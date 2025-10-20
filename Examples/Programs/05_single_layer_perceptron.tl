// ============================================
// Single-Layer Perceptron Example
// ============================================
// This program implements a classic perceptron:
// A single neuron that computes a weighted sum of inputs
// and applies a step function for binary classification
//
// Mathematical form: Y = step(W·X) = step(Σᵢ W[i]*X[i])
//
// The perceptron is the foundation of neural networks,
// invented by Frank Rosenblatt in 1958

// ============================================
// Define Weight Vector W (4 elements)
// ============================================
// Weights represent the importance of each input feature
// These weights would typically be learned during training
// Shape: [4]
//
// Visual representation:
// i=0   i=1   i=2   i=3
// [ 0.5  -0.3  0.8  0.2 ]

W[0] = 0.5
W[1] = -0.3
W[2] = 0.8
W[3] = 0.2

// ============================================
// Define Input Vector X (4 elements)
// ============================================
// Input features for classification
// Shape: [4]
//
// Visual representation:
// i=0  i=1  i=2  i=3
// [ 1.0  2.0  1.5  0.5 ]

X[0] = 1.0
X[1] = 2.0
X[2] = 1.5
X[3] = 0.5

// ============================================
// Compute Perceptron Output
// ============================================
// Step 1: Compute weighted sum (dot product)
//         W[i] X[i] = W[0]*X[0] + W[1]*X[1] + W[2]*X[2] + W[3]*X[3]
//                   = 0.5*1.0 + (-0.3)*2.0 + 0.8*1.5 + 0.2*0.5
//                   = 0.5 - 0.6 + 1.2 + 0.1
//                   = 1.2
//
// Step 2: Apply step function
//         Y = step(1.2) = 1  (since 1.2 > 0)
//
// Interpretation: The perceptron classifies this input as positive (class 1)

Y = step(W[i] X[i])

// ============================================
// Expected Result
// ============================================
// Y = 1
//
// The step function outputs:
// - 1 if weighted sum > 0 (positive class)
// - 0 if weighted sum ≤ 0 (negative class)

// ============================================
// Query the result
// ============================================
Y?  // Should return: 1