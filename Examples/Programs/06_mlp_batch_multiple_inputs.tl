// ============================================
// Multi-Layer Perceptron with Batch Processing
// ============================================
// Process 2 examples simultaneously (i=0 and i=1)

// Example 1 input
X1[0] = 1.0
X1[1] = 0.5

// Example 2 input
X2[0] = 0.3
X2[1] = 1.2

// Shared weights for both examples
W1[0, 0, 0] = 0.5
W1[0, 0, 1] = 0.3
W1[0, 1, 0] = -0.4
W1[0, 1, 1] = 0.6

W2[0, 0, 0] = 0.8
W2[0, 1, 0] = 0.2
W2[0, 0, 1] = -0.3
W2[0, 1, 1] = 0.7

// Process example 1
H1_ex1[0, j] = relu(W1[0, j, k] X1[k])
Y_ex1[0, n] = softmax(W2[0, j, n] H1_ex1[0, j])

// Process example 2
H1_ex2[0, j] = relu(W1[0, j, k] X2[k])
Y_ex2[0, n] = softmax(W2[0, j, n] H1_ex2[0, j])

// Query predictions for both examples
Y_ex1[0, 0]?  // Class 0 probability for example 1
Y_ex1[0, 1]?  // Class 1 probability for example 1
Y_ex2[0, 0]?  // Class 0 probability for example 2
Y_ex2[0, 1]?  // Class 1 probability for example 2