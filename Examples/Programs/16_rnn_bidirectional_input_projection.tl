// ============================================
// Bidirectional RNN with Input Projection
// ============================================
// Processes sequence in both forward and backward directions

// Input dimension: 3
// Hidden dimension: 4
// Sequence length: 3

// ============================================
// Forward RNN Parameters
// ============================================

U_fwd[0, 0] = 0.7
U_fwd[0, 1] = 0.3
U_fwd[0, 2] = 0.2
U_fwd[1, 0] = 0.4
U_fwd[1, 1] = 0.6
U_fwd[1, 2] = 0.1
U_fwd[2, 0] = 0.5
U_fwd[2, 1] = 0.2
U_fwd[2, 2] = 0.8
U_fwd[3, 0] = 0.3
U_fwd[3, 1] = 0.5
U_fwd[3, 2] = 0.4

W_fwd[0, 0] = 0.5
W_fwd[0, 1] = 0.2
W_fwd[0, 2] = 0.1
W_fwd[0, 3] = 0.3
W_fwd[1, 0] = 0.3
W_fwd[1, 1] = 0.6
W_fwd[1, 2] = 0.2
W_fwd[1, 3] = 0.1
W_fwd[2, 0] = 0.2
W_fwd[2, 1] = 0.1
W_fwd[2, 2] = 0.5
W_fwd[2, 3] = 0.4
W_fwd[3, 0] = 0.4
W_fwd[3, 1] = 0.3
W_fwd[3, 2] = 0.2
W_fwd[3, 3] = 0.6

// ============================================
// Backward RNN Parameters
// ============================================

U_bwd[0, 0] = 0.6
U_bwd[0, 1] = 0.4
U_bwd[0, 2] = 0.3
U_bwd[1, 0] = 0.5
U_bwd[1, 1] = 0.5
U_bwd[1, 2] = 0.2
U_bwd[2, 0] = 0.4
U_bwd[2, 1] = 0.3
U_bwd[2, 2] = 0.7
U_bwd[3, 0] = 0.4
U_bwd[3, 1] = 0.6
U_bwd[3, 2] = 0.3

W_bwd[0, 0] = 0.6
W_bwd[0, 1] = 0.3
W_bwd[0, 2] = 0.2
W_bwd[0, 3] = 0.4
W_bwd[1, 0] = 0.4
W_bwd[1, 1] = 0.5
W_bwd[1, 2] = 0.3
W_bwd[1, 3] = 0.2
W_bwd[2, 0] = 0.3
W_bwd[2, 1] = 0.2
W_bwd[2, 2] = 0.6
W_bwd[2, 3] = 0.3
W_bwd[3, 0] = 0.5
W_bwd[3, 1] = 0.4
W_bwd[3, 2] = 0.3
W_bwd[3, 3] = 0.5

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

// Create reversed input for backward pass
Input_rev[k, 0] = Input[k, 2]
Input_rev[k, 1] = Input[k, 1]
Input_rev[k, 2] = Input[k, 0]

// ============================================
// Initial States
// ============================================

State_fwd[0, 0] = 0.0
State_fwd[1, 0] = 0.0
State_fwd[2, 0] = 0.0
State_fwd[3, 0] = 0.0

State_bwd[0, 0] = 0.0
State_bwd[1, 0] = 0.0
State_bwd[2, 0] = 0.0
State_bwd[3, 0] = 0.0

// ============================================
// Forward Pass
// ============================================

Input_proj_fwd[i, t] = U_fwd[i, k] Input[k, t]

State_fwd[i, *t+1] = tanh(
    W_fwd[i, j] State_fwd[j, *t]
  + Input_proj_fwd[i, t]
)

// ============================================
// Backward Pass
// ============================================

Input_proj_bwd[i, t] = U_bwd[i, k] Input_rev[k, t]

State_bwd[i, *t+1] = tanh(
    W_bwd[i, j] State_bwd[j, *t]
  + Input_proj_bwd[i, t]
)

// ============================================
// Combine Bidirectional States
// ============================================

// Concatenate forward and backward final states
// Combined[0:3] = State_fwd[:, *3]
// Combined[4:7] = State_bwd[:, *3]

Combined[0] = State_fwd[0, *3]
Combined[1] = State_fwd[1, *3]
Combined[2] = State_fwd[2, *3]
Combined[3] = State_fwd[3, *3]
Combined[4] = State_bwd[0, *3]
Combined[5] = State_bwd[1, *3]
Combined[6] = State_bwd[2, *3]
Combined[7] = State_bwd[3, *3]

// Output from combined representation
W_out[0] = 0.3
W_out[1] = 0.4
W_out[2] = 0.2
W_out[3] = 0.3
W_out[4] = 0.4
W_out[5] = 0.3
W_out[6] = 0.5
W_out[7] = 0.2

Output = sigmoid(W_out[i] Combined[i])

// ============================================
// Queries
// ============================================

State_fwd[i, *3]?
State_bwd[i, *3]?
Combined[i]?
Output?