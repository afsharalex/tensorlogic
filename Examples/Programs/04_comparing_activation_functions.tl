// ============================================
// Activation Function Comparison
// ============================================
// Demonstrates key differences between activations

// Test input: gradually increasing values
input = [-2.0, -1.0, 0.0, 1.0, 2.0]

// Sigmoid: smooth, bounded (0,1)
sigmoid_out[i] = sigmoid(input[i])
// Expected: [0.119, 0.268, 0.500, 0.731, 0.881]

// Tanh: smooth, zero-centered (-1,1)
tanh_out[i] = tanh(input[i])
// Expected: [-0.964, -0.762, 0.000, 0.762, 0.964]

// ReLU: sharp threshold at 0
relu_out[i] = relu(input[i])
// Expected: [0.0, 0.0, 0.0, 1.0, 2.0]

// Step: binary threshold
step_out[i] = step(input[i])
// Expected: [0, 0, 0, 1, 1]

sigmoid_out?
tanh_out?
relu_out?
step_out?