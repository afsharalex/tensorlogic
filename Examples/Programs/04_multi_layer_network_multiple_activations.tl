// ============================================
// Multi-Layer Network with Various Activations
// ============================================

// Input features
X = [1.5, -0.5, 2.0]

// First layer weights
W1 = [[0.5, -0.3, 0.8],
      [0.2, 0.9, -0.4]]

// Hidden layer with ReLU
H[i, j] = relu(W1[i, j, k] X[k])
// H has shape [2] (2 hidden units)

// Second layer weights
W2 = [[0.6, -0.5],
      [0.3, 0.8],
      [0.1, 0.4]]

// Output layer with different activations

// Binary classification output (sigmoid)
binary_output = sigmoid(W2[0, j] H[j])

// Multi-class output (softmax)
logits[i] = W2[i, j] H[j]
class_probs[i.] = softmax(logits[i])

// Threshold output (step)
threshold_output = step(W2[1, j] H[j])

binary_output?      // Probability for binary classification
class_probs?        // Probability distribution over 3 classes
threshold_output?   // Binary decision