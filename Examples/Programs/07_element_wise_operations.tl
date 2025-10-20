// ============================================
// Combined Arithmetic in Neural Network Layer
// ============================================
// Demonstrates typical layer computation:
// output = activation(weights * input + bias - batch_mean) / std

// Input features
input[0] = 1.0
input[1] = 2.0
input[2] = 3.0

// Weights
weights[0] = 0.5
weights[1] = 0.8
weights[2] = 0.3

// Compute weighted sum
weighted_sum = weights[i] * input[i]
// weighted_sum = 0.5*1.0 + 0.8*2.0 + 0.3*3.0
//              = 0.5 + 1.6 + 0.9 = 3.0

// Bias term
bias = 0.5

// Add bias
pre_activation = weighted_sum + bias  // 3.5

// Batch statistics for normalization
batch_mean = 2.0
batch_std = 1.5

// Normalize
normalized = (pre_activation - batch_mean) / batch_std
// normalized = (3.5 - 2.0) / 1.5 = 1.0

// Apply activation
output = relu(normalized)  // relu(1.0) = 1.0

weighted_sum?      // 3.0
pre_activation?    // 3.5
normalized?        // 1.0
output?            // 1.0