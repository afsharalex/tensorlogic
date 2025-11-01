// Logistic Regression Example
// Binary classification with sigmoid activation
// Learn decision boundary for 2D points

// Training data: 4 points with labels (0 or 1)
// Class 0: points near origin
// Class 1: points far from origin
X1 = [0.5, 0.8, 2.5, 3.0]  // Feature 1
X2 = [0.6, 0.7, 2.8, 3.2]  // Feature 2
Y = [0.0, 0.0, 1.0, 1.0]   // Labels

// Learnable parameters: weights and bias
W1 = [0.1]
W2 = [0.1]
b = [0.0]

// Model: logits = w1*x1 + w2*x2 + b
Logits[i] = W1[0] X1[i] + W2[0] X2[i] + b[0]

// Predictions: apply sigmoid
Predictions[i] = sigmoid(Logits[i])

// Loss: Binary Cross-Entropy
// BCE = -[y*log(p) + (1-y)*log(1-p)]
LogP[i] = log(Predictions[i] + 1e-7)      // Add epsilon for numerical stability
Log1minusP[i] = log(1.0 - Predictions[i] + 1e-7)
Loss_per_sample[i] = -(Y[i] Predictions[i] + (1.0 - Y[i]) Log1minusP[i])
Loss = Loss_per_sample[i]  // Sum

// Optimize
Loss? @minimize(lr=0.1, epochs=200, verbose=true)

// Query learned parameters
W1?
W2?
b?
