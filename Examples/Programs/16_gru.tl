// ============================================
// GRU with Virtual Time Index
// ============================================
// Simpler than LSTM, but still powerful

// Hidden size: 3, Input size: 2

// Reset gate weights
W_r[0, 0] = 0.5
W_r[0, 1] = 0.3
W_r[0, 2] = 0.2
W_r[1, 0] = 0.4
W_r[1, 1] = 0.6
W_r[1, 2] = 0.3
W_r[2, 0] = 0.3
W_r[2, 1] = 0.4
W_r[2, 2] = 0.5

U_r[0, 0] = 0.6
U_r[0, 1] = 0.4
U_r[1, 0] = 0.5
U_r[1, 1] = 0.5
U_r[2, 0] = 0.4
U_r[2, 1] = 0.6

// Update gate weights
W_z[0, 0] = 0.4
W_z[0, 1] = 0.5
W_z[0, 2] = 0.3
W_z[1, 0] = 0.6
W_z[1, 1] = 0.3
W_z[1, 2] = 0.4
W_z[2, 0] = 0.5
W_z[2, 1] = 0.4
W_z[2, 2] = 0.3

U_z[0, 0] = 0.5
U_z[0, 1] = 0.5
U_z[1, 0] = 0.6
U_z[1, 1] = 0.4
U_z[2, 0] = 0.4
U_z[2, 1] = 0.6

// Candidate weights
W_h[0, 0] = 0.6
W_h[0, 1] = 0.3
W_h[0, 2] = 0.4
W_h[1, 0] = 0.4
W_h[1, 1] = 0.7
W_h[1, 2] = 0.2
W_h[2, 0] = 0.5
W_h[2, 1] = 0.4
W_h[2, 2] = 0.6

U_h[0, 0] = 0.7
U_h[0, 1] = 0.3
U_h[1, 0] = 0.5
U_h[1, 1] = 0.6
U_h[2, 0] = 0.6
U_h[2, 1] = 0.5

// Input sequence
Input[0, 0] = 1.0
Input[1, 0] = 0.5
Input[0, 1] = 0.8
Input[1, 1] = 0.6
Input[0, 2] = 0.7
Input[1, 2] = 0.9

// Initial state
h[0, 0] = 0.0
h[1, 0] = 0.0
h[2, 0] = 0.0

// ============================================
// GRU Forward Pass
// ============================================

// Reset gate: determines how much past to forget
r[i, *t+1] = sigmoid(
    W_r[i, j] h[j, *t]
  + U_r[i, k] Input[k, t]
)

// Update gate: determines how much to update
z[i, *t+1] = sigmoid(
    W_z[i, j] h[j, *t]
  + U_z[i, k] Input[k, t]
)

// Candidate hidden state (with reset applied)
h_candidate[i, *t+1] = tanh(
    W_h[i, j] (r[j, *t+1] * h[j, *t])
  + U_h[i, k] Input[k, t]
)

// Final hidden state (interpolate between old and candidate)
h[i, *t+1] = (1.0 - z[i, *t+1]) * h[i, *t]
           + z[i, *t+1] * h_candidate[i, *t+1]

// Queries
h[i, 0]?  // Final state after 3 time steps