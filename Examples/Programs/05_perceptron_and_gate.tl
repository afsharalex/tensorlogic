// ============================================
// Logical AND Gate Implementation
// ============================================
// A perceptron can learn logical functions
// AND truth table:
// X[0] X[1] | Output
//  0    0   |   0
//  0    1   |   0
//  1    0   |   0
//  1    1   |   1

// Weights that implement AND
W[0] = 0.5
W[1] = 0.5

// Bias to set threshold
bias = -0.7

// Test case 1: (0, 0) -> 0
X1[0] = 0.0
X1[1] = 0.0
Y1 = step(W[i] X1[i] + bias)
// Calculation: 0.5*0 + 0.5*0 - 0.7 = -0.7 -> step(-0.7) = 0

// Test case 2: (0, 1) -> 0
X2[0] = 0.0
X2[1] = 1.0
Y2 = step(W[i] X2[i] + bias)
// Calculation: 0.5*0 + 0.5*1 - 0.7 = -0.2 -> step(-0.2) = 0

// Test case 3: (1, 0) -> 0
X3[0] = 1.0
X3[1] = 0.0
Y3 = step(W[i] X3[i] + bias)
// Calculation: 0.5*1 + 0.5*0 - 0.7 = -0.2 -> step(-0.2) = 0

// Test case 4: (1, 1) -> 1
X4[0] = 1.0
X4[1] = 1.0
Y4 = step(W[i] X4[i] + bias)
// Calculation: 0.5*1 + 0.5*1 - 0.7 = 0.3 -> step(0.3) = 1

Y1?  // Should return: 0
Y2?  // Should return: 0
Y3?  // Should return: 0
Y4?  // Should return: 1