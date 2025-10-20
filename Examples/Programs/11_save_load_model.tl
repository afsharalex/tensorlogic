// ============================================
// Saving and Loading Model Checkpoints
// ============================================

// Train a model (simplified)
W[0] = 0.5
W[1] = 0.3
W[2] = 0.4

// Save weights to file
"checkpoints/model_epoch_10.weights" = W[j]

// Later, load weights from checkpoint
W_loaded[j] = "checkpoints/model_epoch_10.weights"

// Continue training or inference with loaded weights
predictions[i] = sigmoid(W_loaded[j] X[i, j])