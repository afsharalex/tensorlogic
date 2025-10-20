// Example 6: Multi-Layer Perceptron (3 layers)
// Input -> Hidden1 -> Hidden2 -> Output

// First hidden layer
H1[i, j] = relu(W1[i, j, k] X[k])

// Second hidden layer
H2[i, m] = relu(W2[i, j, m] H1[i, j])

// Output layer with softmax
Y[i, n] = softmax(W3[i, m, n] H2[i, m])
