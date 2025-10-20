// ============================================
// Multi-Layer Network with Various Activations
// ============================================

// Input features - element-by-element initialization
X[0] = 1.5
X[1] = -0.5
X[2] = 2.0

// First layer weights
W1[0, 0] = 0.5
W1[0, 1] = -0.3
W1[0, 2] = 0.8
W1[1, 0] = 0.2
W1[1, 1] = 0.9
W1[1, 2] = -0.4

// Hidden layer with ReLU
H[i, j] = relu(W1[i, j, k] X[k])

// Second layer weights
W2[0, 0] = 0.6
W2[0, 1] = -0.5
W2[1, 0] = 0.3
W2[1, 1] = 0.8
W2[2, 0] = 0.1
W2[2, 1] = 0.4

// Output layer with different activations

// Binary classification output (sigmoid)
binary_output = sigmoid(W2[0, j] H[j])

// Multi-class output (softmax)
logits[i] = W2[i, j] H[j]
class_probs[i.] = softmax(logits[i])

// Threshold output (step)
threshold_output = step(W2[1, j] H[j])

binary_output?
class_probs?
threshold_output?