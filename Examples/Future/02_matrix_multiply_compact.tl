// ============================================
// Matrix Multiplication - Compact Version
// ============================================

// Define matrices using dense initialization
W = [[1.0, 2.0],
     [3.0, 4.0],
     [5.0, 6.0]]

X = [[1.0, 0.0, 2.0, 1.0],
     [0.0, 1.0, 1.0, 2.0]]

// Perform Einstein summation
Y[i, k] = W[i, j] X[j, k]

// Query result
Y[0, 0]?  // Should return: 1.0
Y[1, 2]?  // Should return: 10.0
Y[2, 3]?  // Should return: 17.0
Y?        // Should return: entire 3x4 result matrix
