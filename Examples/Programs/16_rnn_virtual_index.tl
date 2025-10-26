// ============================================
// Recurrent Neural Network with Virtual Time Index (CORRECTED)
// ============================================
// This program implements an RNN that processes a sequence
// using virtual time index (*t) to avoid allocating memory
// for all time steps
//
// Virtual index *t means successive values overwrite the same memory location

// ============================================
// Network Parameters
// ============================================

// Hidden state dimension: 4
// Input dimension: 4 (CORRECTED: must match hidden dimension)
// Sequence length: 5

// Recurrent weights (hidden-to-hidden)
// Shape: [4, 4]
W[0, 0] = 0.5
W[0, 1] = 0.2
W[0, 2] = 0.1
W[0, 3] = 0.3
W[1, 0] = 0.3
W[1, 1] = 0.6
W[1, 2] = 0.2
W[1, 3] = 0.1
W[2, 0] = 0.2
W[2, 1] = 0.1
W[2, 2] = 0.5
W[2, 3] = 0.4
W[3, 0] = 0.4
W[3, 1] = 0.3
W[3, 2] = 0.2
W[3, 3] = 0.6

// Input weights (input-to-hidden)
// Shape: [4, 4] (CORRECTED: now 4x4, not 4x3)
U[0, 0] = 0.7
U[0, 1] = 0.3
U[0, 2] = 0.2
U[0, 3] = 0.5
U[1, 0] = 0.4
U[1, 1] = 0.6
U[1, 2] = 0.1
U[1, 3] = 0.3
U[2, 0] = 0.5
U[2, 1] = 0.2
U[2, 2] = 0.8
U[2, 3] = 0.4
U[3, 0] = 0.3
U[3, 1] = 0.5
U[3, 2] = 0.4
U[3, 3] = 0.6

// Bias
b[0] = 0.1
b[1] = 0.2
b[2] = 0.1
b[3] = 0.3

// ============================================
// Input Sequence
// ============================================
// Shape: [4, 5] (CORRECTED: 4 features to match hidden dim, 5 time steps)
// Note: t is a REAL index (not virtual) for inputs

// Time step 0
Input[0, 0] = 1.0
Input[1, 0] = 0.5
Input[2, 0] = 0.8
Input[3, 0] = 0.3

// Time step 1
Input[0, 1] = 0.8
Input[1, 1] = 0.6
Input[2, 1] = 0.7
Input[3, 1] = 0.4

// Time step 2
Input[0, 2] = 0.6
Input[1, 2] = 0.9
Input[2, 2] = 0.5
Input[3, 2] = 0.7

// Time step 3
Input[0, 3] = 0.9
Input[1, 3] = 0.4
Input[2, 3] = 0.8
Input[3, 3] = 0.5

// Time step 4
Input[0, 4] = 0.7
Input[1, 4] = 0.7
Input[2, 4] = 0.6
Input[3, 4] = 0.8

// ============================================
// Initial Hidden State
// ============================================
// State at t=0
State[0, 0] = 0.0
State[1, 0] = 0.0
State[2, 0] = 0.0
State[3, 0] = 0.0

// ============================================
// RNN Update Rule (CORRECTED)
// ============================================
// CRITICAL FIX: Use the SAME index 'j' for both terms
// This ensures proper einsum alignment
//
// Einstein summation:
// - W[i, j] * State[j, *t] sums over j
// - U[i, j] * Input[j, t] sums over j (same index!)
// - Both produce vectors of length i (hidden_dim)
// - These can then be added element-wise

State[i, *t+1] = relu(
    W[i, j] State[j, *t]
  + U[i, j] Input[j, t]
  + b[i]
)

// Now the dimensions align correctly:
// W[i, j]: [4, 4]
// State[j, *t]: [4] → W[i,j]*State[j,*t] produces [4] (sum over j)
// U[i, j]: [4, 4]
// Input[j, t]: [4] → U[i,j]*Input[j,t] produces [4] (sum over j)
// Both results have the same shape [4], so addition works!

// ============================================
// Execution Semantics
// ============================================
// This executes as:
//
// t=0:
//   Read Input[:,0] and State[:,0] (initial zeros)
//   Compute: State[:,*1] = relu(W @ State[:,0] + U @ Input[:,0] + b)
//   Overwrites State memory
//
// t=1:
//   Read Input[:,1] and State[:,*1] (from previous iteration)
//   Compute: State[:,*2] = relu(W @ State[:,*1] + U @ Input[:,1] + b)
//   Overwrites State memory
//
// ... continues for t=2,3,4
//
// Final state is at State[:,*5] (stored in same memory location)

// ============================================
// Output Layer
// ============================================
// Map final hidden state to output
W_out[0] = 0.6
W_out[1] = 0.4
W_out[2] = 0.5
W_out[3] = 0.3

// Final output (after all time steps)
// MODE B: After recurrence completes, final state is in State[i, 0]
// (all intermediate states were overwritten in that slot)
Output = sigmoid(W_out[i] State[i, 0])

// ============================================
// Queries
// ============================================

// Query final hidden state
// MODE B: Final state is in slot 0 after recurrence completes
State[0, 0]?
State[1, 0]?
State[2, 0]?
State[3, 0]?

// Query final output
Output?