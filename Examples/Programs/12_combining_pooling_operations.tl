// ============================================
// Pooling Combined with Neural Network Layer
// ============================================

// Input features (3 examples, 4 features each)
Input[0, 0] = 1.0
Input[0, 1] = 2.0
Input[0, 2] = 3.0
Input[0, 3] = 4.0
Input[1, 0] = 2.0
Input[1, 1] = 3.0
Input[1, 2] = 4.0
Input[1, 3] = 5.0
Input[2, 0] = 0.5
Input[2, 1] = 1.5
Input[2, 2] = 2.5
Input[2, 3] = 3.5

// Weights for linear transformation
W[0, 0] = 0.5
W[0, 1] = 0.3
W[0, 2] = 0.2
W[0, 3] = 0.4
W[1, 0] = 0.4
W[1, 1] = 0.6
W[1, 2] = 0.1
W[1, 3] = 0.3

// Transform: creates 3 examples with 2 features each
Hidden[i, k] = relu(W[k, j] Input[i, j])

// Global max pooling: one value per feature across all examples
// For each feature k, find the maximum activation across all examples i
GlobalMax[k] max= Hidden[i, k]

// GlobalMax[0] = max(Hidden[0,0], Hidden[1,0], Hidden[2,0])
// GlobalMax[1] = max(Hidden[0,1], Hidden[1,1], Hidden[2,1])

// Global average pooling
GlobalAvg[k] avg= Hidden[i, k]

// Useful for classification: reduces to fixed-size representation
// regardless of number of input examples

GlobalMax[k]?
GlobalAvg[k]?