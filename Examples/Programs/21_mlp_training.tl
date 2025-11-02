// Multi-Layer Perceptron (MLP) Training
// 2-layer neural network with hidden layer
// XOR problem: learn non-linear decision boundary

// XOR training data
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

// Layer 1: Input (2) -> Hidden (3)
W1 = [[0.5, 0.3, -0.2], [0.2, -0.4, 0.6]]
b1 = [0.1, -0.1, 0.0]

// Layer 2: Hidden (3) -> Output (1)
W2 = [[0.4], [0.3], [-0.5]]
b2 = [0.0]

// Forward pass
// Hidden layer: h = relu(X @ W1 + b1)
Hidden_pre[n, h] = X[n, i] W1[i, h] + b1[h]
Hidden[n, h] = relu(Hidden_pre[n, h])

// Output layer: y = sigmoid(h @ W2 + b2)
Output_pre[n] = Hidden[n, h] W2[h, 0] + b2[0]
Output[n] = sigmoid(Output_pre[n])

// Loss: Mean Squared Error (could also use BCE)
Errors[n] = (Output[n] - Y[n])^2
Loss = Errors[n]

// Initial loss
Loss?

// Train the network
Loss? @minimize(lr=0.5, epochs=1000, verbose=true)

// Query learned parameters
W1?
W2?

// Query predictions
Output?
