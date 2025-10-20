// ============================================
// Simplified Batch Normalization
// ============================================
// Common technique in deep learning to stabilize training
// Normalizes a batch of activations

// Batch of 4 activations
activations[0] = 3.0
activations[1] = 5.0
activations[2] = 7.0
activations[3] = 9.0

// Compute batch mean
// batch_mean = (3 + 5 + 7 + 9) / 4 = 6.0
batch_mean = activations[i] / 4.0

// Compute batch variance (simplified - assume we calculated it)
batch_var = 5.0  // For demonstration

// Small constant for numerical stability
epsilon = 0.001

// Normalize
// normalized[i] = (activations[i] - batch_mean) / sqrt(batch_var + epsilon)
denominator = sqrt(batch_var + epsilon)

// normalized[0] = (3.0 - 6.0) / sqrt(5.001) ≈ -1.341
// normalized[1] = (5.0 - 6.0) / sqrt(5.001) ≈ -0.447
// normalized[2] = (7.0 - 6.0) / sqrt(5.001) ≈ 0.447
// normalized[3] = (9.0 - 6.0) / sqrt(5.001) ≈ 1.341
normalized[i] = (activations[i] - batch_mean) / denominator

// Scale and shift parameters (learned during training)
gamma = 2.0
beta = 1.0

// Final batch-normalized output
// Y[i] = gamma * normalized[i] + beta
Y_bn[i] = gamma * normalized[i] + beta

normalized[0]?  // ~-1.341
Y_bn[0]?        // ~-1.682