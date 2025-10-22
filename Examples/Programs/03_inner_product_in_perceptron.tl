// ============================================
// Simple Perceptron using Inner Product
// ============================================
// A perceptron computes: output = activation(W·X + bias)

// Weights
W = [0.5, -0.3, 0.8]

// Input features
X = [1.0, 2.0, 1.5]

// Bias term
bias = 0.2

// Compute weighted sum
weighted_sum = W[i] X[i] + bias

// Apply step activation function
output = step(weighted_sum)

// Step-by-step:
// weighted_sum = 0.5*1.0 + (-0.3)*2.0 + 0.8*1.5 + 0.2
//              = 0.5 - 0.6 + 1.2 + 0.2
//              = 1.3
// output = step(1.3) = 1.0 (since 1.3 > 0)

weighted_sum?  // Expected: 1.3
output?        // Expected: 1.0