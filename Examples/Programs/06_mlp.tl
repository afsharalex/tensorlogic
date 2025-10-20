// ============================================
// Multi-Layer Perceptron (3 layers)
// ============================================
// Architecture: Input(3) -> Hidden1(4) -> Hidden2(2) -> Output(3)
//
// This demonstrates a deep neural network with:
// - Input layer: 3 features
// - First hidden layer: 4 neurons with ReLU activation
// - Second hidden layer: 2 neurons with ReLU activation
// - Output layer: 3 classes with softmax activation
//
// Note: The index 'i' represents the example/batch dimension
//       We'll use i=0 for a single example

// ============================================
// Define Input Vector X (3 features)
// ============================================
// Shape: [3]
// Sample input for classification (e.g., flower measurements)
//
// Visual representation:
// k=0  k=1  k=2
// [ 1.5  0.8  2.1 ]

X[0] = 1.5
X[1] = 0.8
X[2] = 2.1

// ============================================
// First Hidden Layer Weights W1
// ============================================
// Shape: [1, 4, 3] (batch_size=1, hidden1_units=4, input_features=3)
// Maps 3 input features to 4 hidden units
//
// Visual representation (for i=0):
//        k=0   k=1   k=2
// j=0  [ 0.5   0.3  -0.2 ]
// j=1  [-0.4   0.6   0.1 ]
// j=2  [ 0.7  -0.3   0.5 ]
// j=3  [ 0.2   0.4  -0.6 ]

// Hidden unit 0 weights
W1[0, 0, 0] = 0.5
W1[0, 0, 1] = 0.3
W1[0, 0, 2] = -0.2

// Hidden unit 1 weights
W1[0, 1, 0] = -0.4
W1[0, 1, 1] = 0.6
W1[0, 1, 2] = 0.1

// Hidden unit 2 weights
W1[0, 2, 0] = 0.7
W1[0, 2, 1] = -0.3
W1[0, 2, 2] = 0.5

// Hidden unit 3 weights
W1[0, 3, 0] = 0.2
W1[0, 3, 1] = 0.4
W1[0, 3, 2] = -0.6

// ============================================
// Compute First Hidden Layer
// ============================================
// H1[i, j] = relu(W1[i, j, k] X[k])
// For each hidden unit j, compute weighted sum of inputs and apply ReLU
//
// Expected calculations (for i=0):
// H1[0, 0] = relu(0.5*1.5 + 0.3*0.8 + (-0.2)*2.1)
//          = relu(0.75 + 0.24 - 0.42)
//          = relu(0.57) = 0.57
//
// H1[0, 1] = relu(-0.4*1.5 + 0.6*0.8 + 0.1*2.1)
//          = relu(-0.6 + 0.48 + 0.21)
//          = relu(0.09) = 0.09
//
// H1[0, 2] = relu(0.7*1.5 + (-0.3)*0.8 + 0.5*2.1)
//          = relu(1.05 - 0.24 + 1.05)
//          = relu(1.86) = 1.86
//
// H1[0, 3] = relu(0.2*1.5 + 0.4*0.8 + (-0.6)*2.1)
//          = relu(0.3 + 0.32 - 1.26)
//          = relu(-0.64) = 0.0 (ReLU zeros negative values)

H1[i, j] = relu(W1[i, j, k] X[k])

// ============================================
// Second Hidden Layer Weights W2
// ============================================
// Shape: [1, 4, 2] (batch_size=1, hidden1_units=4, hidden2_units=2)
// Maps 4 first-layer hidden units to 2 second-layer hidden units
//
// Visual representation (for i=0):
//        j=0   j=1   j=2   j=3
// m=0  [ 0.8  -0.3   0.4   0.6 ]
// m=1  [ 0.2   0.5  -0.7   0.3 ]

// Second hidden unit 0 weights
W2[0, 0, 0] = 0.8
W2[0, 1, 0] = -0.3
W2[0, 2, 0] = 0.4
W2[0, 3, 0] = 0.6

// Second hidden unit 1 weights
W2[0, 0, 1] = 0.2
W2[0, 1, 1] = 0.5
W2[0, 2, 1] = -0.7
W2[0, 3, 1] = 0.3

// ============================================
// Compute Second Hidden Layer
// ============================================
// H2[i, m] = relu(W2[i, j, m] H1[i, j])
// For each hidden unit m, compute weighted sum of H1 activations
//
// Expected calculations (for i=0):
// H2[0, 0] = relu(0.8*0.57 + (-0.3)*0.09 + 0.4*1.86 + 0.6*0.0)
//          = relu(0.456 - 0.027 + 0.744 + 0.0)
//          = relu(1.173) = 1.173
//
// H2[0, 1] = relu(0.2*0.57 + 0.5*0.09 + (-0.7)*1.86 + 0.3*0.0)
//          = relu(0.114 + 0.045 - 1.302 + 0.0)
//          = relu(-1.143) = 0.0

H2[i, m] = relu(W2[i, j, m] H1[i, j])

// ============================================
// Output Layer Weights W3
// ============================================
// Shape: [1, 2, 3] (batch_size=1, hidden2_units=2, output_classes=3)
// Maps 2 second-layer hidden units to 3 output classes
//
// Visual representation (for i=0):
//        m=0   m=1
// n=0  [ 0.9  -0.4 ]
// n=1  [ 0.3   0.7 ]
// n=2  [-0.5   0.6 ]

// Output class 0 weights
W3[0, 0, 0] = 0.9
W3[0, 1, 0] = -0.4

// Output class 1 weights
W3[0, 0, 1] = 0.3
W3[0, 1, 1] = 0.7

// Output class 2 weights
W3[0, 0, 2] = -0.5
W3[0, 1, 2] = 0.6

// ============================================
// Compute Output Layer with Softmax
// ============================================
// Y[i, n] = softmax(W3[i, m, n] H2[i, m])
// First compute logits, then apply softmax for probability distribution
//
// Expected calculations (for i=0):
// Logit[0] = 0.9*1.173 + (-0.4)*0.0 = 1.0557
// Logit[1] = 0.3*1.173 + 0.7*0.0 = 0.3519
// Logit[2] = -0.5*1.173 + 0.6*0.0 = -0.5865
//
// Softmax normalization:
// exp(1.0557) ≈ 2.874
// exp(0.3519) ≈ 1.422
// exp(-0.5865) ≈ 0.556
// Sum ≈ 4.852
//
// Y[0, 0] = 2.874 / 4.852 ≈ 0.592 (59.2% probability for class 0)
// Y[0, 1] = 1.422 / 4.852 ≈ 0.293 (29.3% probability for class 1)
// Y[0, 2] = 0.556 / 4.852 ≈ 0.115 (11.5% probability for class 2)
//
// The notation Y[i, n.] would indicate softmax normalization over n
// but the paper example uses Y[i, n] with softmax() function

Y[i, n] = softmax(W3[i, m, n] H2[i, m])

// ============================================
// Query Results
// ============================================

// Query intermediate hidden layer activations
H1[0, 0]?  // Should return: 0.57
H1[0, 1]?  // Should return: 0.09
H1[0, 2]?  // Should return: 1.86
H1[0, 3]?  // Should return: 0.0

H2[0, 0]?  // Should return: 1.173
H2[0, 1]?  // Should return: 0.0

// Query output probabilities
Y[0, 0]?  // Should return: ~0.592 (class 0 probability)
Y[0, 1]?  // Should return: ~0.293 (class 1 probability)
Y[0, 2]?  // Should return: ~0.115 (class 2 probability)

// Query entire output distribution
Y[0, n]?  // Should return: probability distribution over 3 classes

// ============================================
// Verification: Softmax probabilities sum to 1
// ============================================
prob_sum = Y[0, n]
prob_sum?  // Should return: 1.0 (or very close)