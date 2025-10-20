// ============================================
// Perceptron with Bias - More Realistic
// ============================================
// Most perceptrons include a bias term for threshold adjustment
// Form: Y = step(W·X + b)

// Weights
W[0] = 0.4
W[1] = 0.6
W[2] = -0.3

// Inputs
X[0] = 1.0
X[1] = 0.5
X[2] = 2.0

// Bias term (shifts the decision boundary)
bias = -0.5

// Compute weighted sum with bias
// weighted_sum = W[i]*X[i] + bias
//              = 0.4*1.0 + 0.6*0.5 + (-0.3)*2.0 + (-0.5)
//              = 0.4 + 0.3 - 0.6 - 0.5
//              = -0.4
// Y = step(-0.4) = 0

Y = step(W[i] X[i] + bias)

Y?  // Should return: 0