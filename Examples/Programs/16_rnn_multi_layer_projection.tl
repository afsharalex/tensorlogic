// ============================================
// Two-Layer RNN with Input Projection
// ============================================
// Demonstrates stacking RNN layers

// Input dimension: 3
// Hidden dimension layer 1: 4
// Hidden dimension layer 2: 3
// Sequence length: 4

// ============================================
// Layer 1 Parameters
// ============================================

// Input projection: 3 → 4
U1[0, 0] = 0.7
U1[0, 1] = 0.3
U1[0, 2] = 0.2
U1[1, 0] = 0.4
U1[1, 1] = 0.6
U1[1, 2] = 0.1
U1[2, 0] = 0.5
U1[2, 1] = 0.2
U1[2, 2] = 0.8
U1[3, 0] = 0.3
U1[3, 1] = 0.5
U1[3, 2] = 0.4

// Recurrent weights layer 1: 4 × 4
W1[0, 0] = 0.5
W1[0, 1] = 0.2
W1[0, 2] = 0.1
W1[0, 3] = 0.3
W1[1, 0] = 0.3
W1[1, 1] = 0.6
W1[1, 2] = 0.2
W1[1, 3] = 0.1
W1[2, 0] = 0.2
W1[2, 1] = 0.1
W1[2, 2] = 0.5
W1[2, 3] = 0.4
W1[3, 0] = 0.4
W1[3, 1] = 0.3
W1[3, 2] = 0.2
W1[3, 3] = 0.6

b1[0] = 0.1
b1[1] = 0.2
b1[2] = 0.1
b1[3] = 0.3

// ============================================
// Layer 2 Parameters
// ============================================

// Input projection for layer 2: 4 → 3
U2[0, 0] = 0.6
U2[0, 1] = 0.3
U2[0, 2] = 0.4
U2[0, 3] = 0.2
U2[1, 0] = 0.5
U2[1, 1] = 0.4
U2[1, 2] = 0.3
U2[1, 3] = 0.5
U2[2, 0] = 0.4
U2[2, 1] = 0.5
U2[2, 2] = 0.6
U2[2, 3] = 0.3

// Recurrent weights layer 2: 3 × 3
W2[0, 0] = 0.6
W2[0, 1] = 0.2
W2[0, 2] = 0.3
W2[1, 0] = 0.3
W2[1, 1] = 0.7
W2[1, 2] = 0.2
W2[2, 0] = 0.4
W2[2, 1] = 0.3
W2[2, 2] = 0.6

b2[0] = 0.2
b2[1] = 0.1
b2[2] = 0.3

// ============================================
// Input Sequence
// ============================================

Input[0, 0] = 1.0
Input[1, 0] = 0.5
Input[2, 0] = 0.8

Input[0, 1] = 0.8
Input[1, 1] = 0.6
Input[2, 1] = 0.7

Input[0, 2] = 0.6
Input[1, 2] = 0.9
Input[2, 2] = 0.5

Input[0, 3] = 0.9
Input[1, 3] = 0.4
Input[2, 3] = 0.8

// ============================================
// Initial States
// ============================================

State1[0, 0] = 0.0
State1[1, 0] = 0.0
State1[2, 0] = 0.0
State1[3, 0] = 0.0

State2[0, 0] = 0.0
State2[1, 0] = 0.0
State2[2, 0] = 0.0

// ============================================
// Layer 1: Process Input
// ============================================

// Project input
Input_proj1[i, t] = U1[i, k] Input[k, t]

// RNN layer 1
State1[i, *t+1] = relu(
    W1[i, j] State1[j, *t]
  + Input_proj1[i, t]
  + b1[i]
)

// ============================================
// Layer 2: Process Layer 1 Output
// ============================================

// Project layer 1 output to layer 2 input
// Note: State1[j, *t+1] is the output of layer 1 at time t+1
// We need to use this as input to layer 2
Input_proj2[i, t] = U2[i, j] State1[j, *t+1]

// RNN layer 2
State2[i, *t+1] = relu(
    W2[i, j] State2[j, *t]
  + Input_proj2[i, t]
  + b2[i]
)

// ============================================
// Output Layer
// ============================================

W_out[0] = 0.5
W_out[1] = 0.4
W_out[2] = 0.6

Output = sigmoid(W_out[i] State2[i, *4])

// ============================================
// Queries
// ============================================

// Layer 1 final state
State1[i, *4]?

// Layer 2 final state
State2[i, *4]?

// Final output
Output?