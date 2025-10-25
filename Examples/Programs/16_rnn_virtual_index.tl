// ============================================
// Recurrent Neural Network with Virtual Time Index
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
// Input dimension: 3
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
// Shape: [4, 3]
U[0, 0] = 0.7
U[0, 1] = 0.3
U[0, 2] = 0.2
U[1, 0] = 0.4
U[1, 1] = 0.6
U[1, 2] = 0.1
U[2, 0] = 0.5
U[2, 1] = 0.2
U[2, 2] = 0.8
U[3, 0] = 0.3
U[3, 1] = 0.5
U[3, 2] = 0.4

// Bias
b[0] = 0.1
b[1] = 0.2
b[2] = 0.1
b[3] = 0.3

// ============================================
// Input Sequence
// ============================================
// Shape: [3, 5] (3 features, 5 time steps)
// Note: t is a REAL index (not virtual) for inputs

// Time step 0
Input[0, 0] = 1.0
Input[1, 0] = 0.5
Input[2, 0] = 0.8

// Time step 1
Input[0, 1] = 0.8
Input[1, 1] = 0.6
Input[2, 1] = 0.7

// Time step 2
Input[0, 2] = 0.6
Input[1, 2] = 0.9
Input[2, 2] = 0.5

// Time step 3
Input[0, 3] = 0.9
Input[1, 3] = 0.4
Input[2, 3] = 0.8

// Time step 4
Input[0, 4] = 0.7
Input[1, 4] = 0.7
Input[2, 4] = 0.6

// ============================================
// Initial Hidden State
// ============================================
// State at t=0
State[0, 0] = 0.0
State[1, 0] = 0.0
State[2, 0] = 0.0
State[3, 0] = 0.0

// ============================================
// RNN Update Rule
// ============================================
// Virtual index *t+1 means:
// - Read from State[j, *t] (previous time step)
// - Write to State[i, *t+1] (current time step)
// - Both use the SAME memory location
// - No memory allocation for full sequence
//
// At each iteration:
// 1. Compute recurrent contribution: W * State[*t]
// 2. Compute input contribution: U * Input[t]
// 3. Add bias and apply ReLU
// 4. Overwrite State with new values

State[i, *t+1] = relu(
    W[i, j] State[j, *t]
  + U[i, k] Input[k, t]
  + b[i]
)

// ============================================
// Execution Semantics
// ============================================
// This executes as:
//
// t=0: Read Input[:,0], State[:,0] (initial)
//      Write to State[:, *1] (overwrites State[:, 0])
//
// t=1: Read Input[:,1], State[:, *1] (from previous)
//      Write to State[:, *2] (overwrites State[:, 1])
//
// ... continues for t=2,3,4
//
// Final state is at State[:, *5] (stored in same location)

// ============================================
// Output Layer (Optional)
// ============================================
// Map final hidden state to output
W_out[0] = 0.6
W_out[1] = 0.4
W_out[2] = 0.5
W_out[3] = 0.3

// Final output (after all time steps)
// Uses the final state stored at virtual time *5
Output = sigmoid(W_out[i] State[i, *5])

// ============================================
// Queries
// ============================================

// Query final hidden state
// Note: State[i, *5] contains the final state
// (all intermediate states were overwritten)
State[0, *5]?
State[1, *5]?
State[2, *5]?
State[3, *5]?

// Query final output
Output?