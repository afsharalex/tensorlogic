// ============================================
// Perceptron - Negative Classification Example
// ============================================
// Same perceptron, different input that results in negative classification

// Weights (same as before)
W[0] = 0.5
W[1] = -0.3
W[2] = 0.8
W[3] = 0.2

// Different input that will be classified as negative
X[0] = 0.0
X[1] = 3.0
X[2] = 0.2
X[3] = 0.1

// Compute output
// Weighted sum = 0.5*0.0 + (-0.3)*3.0 + 0.8*0.2 + 0.2*0.1
//              = 0.0 - 0.9 + 0.16 + 0.02
//              = -0.72
// Y = step(-0.72) = 0 (negative class)

Y = step(W[i] X[i])

Y?  // Should return: 0