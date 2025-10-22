// ============================================
// Activation Functions - Compact Version
// ============================================

// Define input vector
X = [-3.0, -1.0, -0.5, 0.0, 0.5, 2.0, 5.0]

// Apply activation functions
Y1[i] = sigmoid(X[i])  // Sigmoid: maps to (0,1)
Y2[i] = relu(X[i])     // ReLU: max(0,x)
Y3[i] = tanh(X[i])     // Tanh: maps to (-1,1)
Y4[i.] = softmax(X[i]) // Softmax: probability distribution
Y5[i] = step(X[i])     // Step: binary threshold

// Query all results
Y1?
Y2?
Y3?
Y4?
Y5?