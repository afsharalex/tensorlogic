// ============================================
// Simplified Multi-Layer Perceptron (2 layers)
// ============================================
// Architecture: Input(2) -> Hidden(2) -> Output(2)
// Easier to verify calculations by hand

// Input
X[0] = 1.0
X[1] = 0.5

// First layer weights (2x2 matrix)
W1[0, 0, 0] = 0.5
W1[0, 0, 1] = 0.3
W1[0, 1, 0] = -0.4
W1[0, 1, 1] = 0.6

// Hidden layer computation
// H1[0, 0] = relu(0.5*1.0 + 0.3*0.5) = relu(0.65) = 0.65
// H1[0, 1] = relu(-0.4*1.0 + 0.6*0.5) = relu(-0.1) = 0.0
H1[i, j] = relu(W1[i, j, k] X[k])

// Output layer weights (2x2 matrix)
W2[0, 0, 0] = 0.8
W2[0, 1, 0] = 0.2
W2[0, 0, 1] = -0.3
W2[0, 1, 1] = 0.7

// Output layer with softmax
// Logit[0] = 0.8*0.65 + 0.2*0.0 = 0.52
// Logit[1] = -0.3*0.65 + 0.7*0.0 = -0.195
// After softmax normalization:
// Y[0, 0] ≈ 0.672 (67.2%)
// Y[0, 1] ≈ 0.328 (32.8%)
Y[i, n] = softmax(W2[i, j, n] H1[i, j])

H1[0, 0]?  // 0.65
H1[0, 1]?  // 0.0
Y[0, 0]?   // ~0.672
Y[0, 1]?   // ~0.328