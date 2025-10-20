// ============================================
// Activation Functions Example
// ============================================
// This program demonstrates various nonlinear activation functions
// commonly used in neural networks, applied elementwise to a vector

// ============================================
// Define Input Vector X (7 elements)
// ============================================
// X contains a range of values to demonstrate each activation's behavior:
// - Negative values (to show how each function handles them)
// - Zero (boundary case)
// - Positive values (normal activation range)
// - Large positive value (saturation behavior)
//
// Shape: [7]
//
// Visual representation:
// i=0    i=1   i=2  i=3  i=4  i=5  i=6
// [-3.0  -1.0  -0.5  0.0  0.5  2.0  5.0]

X[0] = -3.0
X[1] = -1.0
X[2] = -0.5
X[3] = 0.0
X[4] = 0.5
X[5] = 2.0
X[6] = 5.0

// ============================================
// Activation Function 1: Sigmoid
// ============================================
// Sigmoid(x) = 1 / (1 + e^(-x))
// - Range: (0, 1)
// - Smooth S-shaped curve
// - Used for binary classification, gating mechanisms
// - Saturates for large |x| (gradient vanishing problem)
//
// Expected outputs for our X values:
// Y1[0] = sigmoid(-3.0) ≈ 0.047  (very close to 0)
// Y1[1] = sigmoid(-1.0) ≈ 0.268
// Y1[2] = sigmoid(-0.5) ≈ 0.378
// Y1[3] = sigmoid(0.0)  = 0.500  (midpoint)
// Y1[4] = sigmoid(0.5)  ≈ 0.622
// Y1[5] = sigmoid(2.0)  ≈ 0.881
// Y1[6] = sigmoid(5.0)  ≈ 0.993  (very close to 1)

Y1[i] = sigmoid(X[i])

// ============================================
// Activation Function 2: ReLU
// ============================================
// ReLU(x) = max(0, x)
// - Range: [0, ∞)
// - Piecewise linear: 0 for x<0, identity for x≥0
// - Most popular activation in deep learning
// - Computationally efficient
// - Helps mitigate vanishing gradient problem
//
// Expected outputs for our X values:
// Y2[0] = relu(-3.0) = 0.0  (negative → zero)
// Y2[1] = relu(-1.0) = 0.0  (negative → zero)
// Y2[2] = relu(-0.5) = 0.0  (negative → zero)
// Y2[3] = relu(0.0)  = 0.0  (boundary case)
// Y2[4] = relu(0.5)  = 0.5  (positive → unchanged)
// Y2[5] = relu(2.0)  = 2.0  (positive → unchanged)
// Y2[6] = relu(5.0)  = 5.0  (positive → unchanged)

Y2[i] = relu(X[i])

// ============================================
// Activation Function 3: Tanh
// ============================================
// Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
// - Range: (-1, 1)
// - Zero-centered (unlike sigmoid)
// - Smooth S-shaped curve
// - Used in RNNs, when zero-centered outputs are desired
// - Saturates for large |x|
//
// Expected outputs for our X values:
// Y3[0] = tanh(-3.0) ≈ -0.995  (very close to -1)
// Y3[1] = tanh(-1.0) ≈ -0.762
// Y3[2] = tanh(-0.5) ≈ -0.462
// Y3[3] = tanh(0.0)  = 0.000   (zero-centered)
// Y3[4] = tanh(0.5)  ≈ 0.462
// Y3[5] = tanh(2.0)  ≈ 0.964
// Y3[6] = tanh(5.0)  ≈ 0.999   (very close to 1)

Y3[i] = tanh(X[i])

// ============================================
// Activation Function 4: Softmax
// ============================================
// Softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
// - Range: (0, 1) for each element
// - Sum of all outputs = 1 (probability distribution)
// - Used for multi-class classification
// - Normalizes across the entire vector
//
// The notation Y4[i.] indicates that normalization is performed
// over the index 'i' (the dot indicates which index to normalize)
//
// Expected outputs (sum must equal 1.0):
// First compute: sum = e^(-3.0) + e^(-1.0) + e^(-0.5) + e^(0.0) + e^(0.5) + e^(2.0) + e^(5.0)
//                    ≈ 0.050 + 0.368 + 0.606 + 1.000 + 1.649 + 7.389 + 148.413
//                    ≈ 159.475
//
// Y4[0] = e^(-3.0) / 159.475 ≈ 0.0003  (smallest probability)
// Y4[1] = e^(-1.0) / 159.475 ≈ 0.0023
// Y4[2] = e^(-0.5) / 159.475 ≈ 0.0038
// Y4[3] = e^(0.0)  / 159.475 ≈ 0.0063
// Y4[4] = e^(0.5)  / 159.475 ≈ 0.0103
// Y4[5] = e^(2.0)  / 159.475 ≈ 0.0463
// Y4[6] = e^(5.0)  / 159.475 ≈ 0.9307  (largest probability - dominates!)

Y4[i.] = softmax(X[i])

// ============================================
// Activation Function 5: Step (Heaviside)
// ============================================
// Step(x) = 1 if x > 0, else 0
// - Range: {0, 1}
// - Non-differentiable (problematic for backpropagation)
// - Used in classical perceptrons
// - Threshold/binary activation
// - The paper uses H(x) and step(x) interchangeably
//
// Expected outputs for our X values:
// Y5[0] = step(-3.0) = 0  (negative → 0)
// Y5[1] = step(-1.0) = 0  (negative → 0)
// Y5[2] = step(-0.5) = 0  (negative → 0)
// Y5[3] = step(0.0)  = 0  (boundary: typically 0)
// Y5[4] = step(0.5)  = 1  (positive → 1)
// Y5[5] = step(2.0)  = 1  (positive → 1)
// Y5[6] = step(5.0)  = 1  (positive → 1)

Y5[i] = step(X[i])

// ============================================
// Query Individual Results
// ============================================

// Query specific elements to verify computations
X[3]?   // Should return: 0.0
Y1[3]?  // Sigmoid(0.0) = 0.5
Y2[0]?  // ReLU(-3.0) = 0.0
Y3[3]?  // Tanh(0.0) = 0.0
Y4[6]?  // Softmax: largest value dominates ≈ 0.9307
Y5[4]?  // Step(0.5) = 1.0

// Query entire activation outputs
Y1?  // Sigmoid output vector
Y2?  // ReLU output vector
Y3?  // Tanh output vector
Y4?  // Softmax output vector (sums to 1.0)
Y5?  // Step output vector (binary)

// ============================================
// Verification: Softmax sum should equal 1.0
// ============================================
softmax_sum = Y4[i]
softmax_sum?  // Should return: 1.0 (or very close due to floating point)