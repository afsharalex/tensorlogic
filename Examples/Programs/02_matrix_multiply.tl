// ============================================
// Matrix Multiplication Example
// ============================================
// This program multiplies a 3x2 matrix W by a 2x4 matrix X
// to produce a 3x4 matrix Y using Einstein summation
// Y[i,k] = sum over j of W[i,j] * X[j,k]

// ============================================
// Define Input Matrix W (3 rows, 2 columns)
// ============================================
// W represents a weight matrix
// Shape: [3, 2]
//
// Visual representation:
//     j=0  j=1
// i=0 [ 1.0  2.0 ]
// i=1 [ 3.0  4.0 ]
// i=2 [ 5.0  6.0 ]

W[0, 0] = 1.0
W[0, 1] = 2.0
W[1, 0] = 3.0
W[1, 1] = 4.0
W[2, 0] = 5.0
W[2, 1] = 6.0

// ============================================
// Define Input Matrix X (2 rows, 4 columns)
// ============================================
// X represents an input/feature matrix
// Shape: [2, 4]
//
// Visual representation:
//     k=0  k=1  k=2  k=3
// j=0 [ 1.0  0.0  2.0  1.0 ]
// j=1 [ 0.0  1.0  1.0  2.0 ]

X[0, 0] = 1.0
X[0, 1] = 0.0
X[0, 2] = 2.0
X[0, 3] = 1.0
X[1, 0] = 0.0
X[1, 1] = 1.0
X[1, 2] = 1.0
X[1, 3] = 2.0

// ============================================
// Perform Matrix Multiplication
// ============================================
// Einstein summation: Y[i,k] = W[i,j] * X[j,k]
// - Index 'i' appears in Y and W (rows of result and W)
// - Index 'k' appears in Y and X (columns of result and X)
// - Index 'j' appears in both W and X but NOT in Y
//   (this is the summed/contracted dimension)
//
// The implicit summation over j means:
// Y[i,k] = W[i,0]*X[0,k] + W[i,1]*X[1,k] + ... + W[i,n]*X[n,k]

Y[i, k] = W[i, j] X[j, k]

// ============================================
// Expected Result: Y (3 rows, 4 columns)
// ============================================
// Shape: [3, 4]
//
// Computed as:
// Y[0,0] = W[0,0]*X[0,0] + W[0,1]*X[1,0] = 1.0*1.0 + 2.0*0.0 = 1.0
// Y[0,1] = W[0,0]*X[0,1] + W[0,1]*X[1,1] = 1.0*0.0 + 2.0*1.0 = 2.0
// Y[0,2] = W[0,0]*X[0,2] + W[0,1]*X[1,2] = 1.0*2.0 + 2.0*1.0 = 4.0
// Y[0,3] = W[0,0]*X[0,3] + W[0,1]*X[1,3] = 1.0*1.0 + 2.0*2.0 = 5.0
//
// Y[1,0] = W[1,0]*X[0,0] + W[1,1]*X[1,0] = 3.0*1.0 + 4.0*0.0 = 3.0
// Y[1,1] = W[1,0]*X[0,1] + W[1,1]*X[1,1] = 3.0*0.0 + 4.0*1.0 = 4.0
// Y[1,2] = W[1,0]*X[0,2] + W[1,1]*X[1,2] = 3.0*2.0 + 4.0*1.0 = 10.0
// Y[1,3] = W[1,0]*X[0,3] + W[1,1]*X[1,3] = 3.0*1.0 + 4.0*2.0 = 11.0
//
// Y[2,0] = W[2,0]*X[0,0] + W[2,1]*X[1,0] = 5.0*1.0 + 6.0*0.0 = 5.0
// Y[2,1] = W[2,0]*X[0,1] + W[2,1]*X[1,1] = 5.0*0.0 + 6.0*1.0 = 6.0
// Y[2,2] = W[2,0]*X[0,2] + W[2,1]*X[1,2] = 5.0*2.0 + 6.0*1.0 = 16.0
// Y[2,3] = W[2,0]*X[0,3] + W[2,1]*X[1,3] = 5.0*1.0 + 6.0*2.0 = 17.0
//
// Final result matrix Y:
//     k=0  k=1   k=2   k=3
// i=0 [ 1.0  2.0   4.0   5.0 ]
// i=1 [ 3.0  4.0  10.0  11.0 ]
// i=2 [ 5.0  6.0  16.0  17.0 ]

// ============================================
// Query the result
// ============================================
// To inspect the computed values, we can query specific elements
// or the entire tensor

Y[0, 0]?  // Should return: 1.0
Y[1, 2]?  // Should return: 10.0
Y[2, 3]?  // Should return: 17.0
Y?        // Should return: entire 3x4 result matrix
