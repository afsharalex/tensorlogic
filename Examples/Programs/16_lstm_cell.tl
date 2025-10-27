// ============================================
// LSTM (Long Short-Term Memory) with Virtual Indices
// ============================================
// Implements LSTM cell with forget, input, and output gates
// using virtual time index to minimize memory usage

// ============================================
// Parameters
// ============================================
// Hidden size: 3
// Input size: 2
// Sequence length: 4

// Forget gate weights
W_f[0, 0] = 0.5
W_f[0, 1] = 0.3
W_f[0, 2] = 0.2
W_f[1, 0] = 0.4
W_f[1, 1] = 0.6
W_f[1, 2] = 0.1
W_f[2, 0] = 0.3
W_f[2, 1] = 0.2
W_f[2, 2] = 0.5

U_f[0, 0] = 0.6
U_f[0, 1] = 0.4
U_f[1, 0] = 0.5
U_f[1, 1] = 0.3
U_f[2, 0] = 0.4
U_f[2, 1] = 0.6

// Input gate weights
W_i[0, 0] = 0.4
W_i[0, 1] = 0.5
W_i[0, 2] = 0.3
W_i[1, 0] = 0.6
W_i[1, 1] = 0.2
W_i[1, 2] = 0.4
W_i[2, 0] = 0.3
W_i[2, 1] = 0.5
W_i[2, 2] = 0.2

U_i[0, 0] = 0.5
U_i[0, 1] = 0.5
U_i[1, 0] = 0.4
U_i[1, 1] = 0.6
U_i[2, 0] = 0.6
U_i[2, 1] = 0.3

// Cell candidate weights
W_c[0, 0] = 0.6
W_c[0, 1] = 0.2
W_c[0, 2] = 0.4
W_c[1, 0] = 0.3
W_c[1, 1] = 0.7
W_c[1, 2] = 0.2
W_c[2, 0] = 0.5
W_c[2, 1] = 0.3
W_c[2, 2] = 0.6

U_c[0, 0] = 0.7
U_c[0, 1] = 0.3
U_c[1, 0] = 0.4
U_c[1, 1] = 0.8
U_c[2, 0] = 0.6
U_c[2, 1] = 0.5

// Output gate weights
W_o[0, 0] = 0.5
W_o[0, 1] = 0.4
W_o[0, 2] = 0.3
W_o[1, 0] = 0.6
W_o[1, 1] = 0.3
W_o[1, 2] = 0.5
W_o[2, 0] = 0.4
W_o[2, 1] = 0.6
W_o[2, 2] = 0.4

U_o[0, 0] = 0.6
U_o[0, 1] = 0.5
U_o[1, 0] = 0.5
U_o[1, 1] = 0.4
U_o[2, 0] = 0.7
U_o[2, 1] = 0.4

// ============================================
// Input Sequence
// ============================================
Input[0, 0] = 1.0
Input[1, 0] = 0.5

Input[0, 1] = 0.8
Input[1, 1] = 0.7

Input[0, 2] = 0.6
Input[1, 2] = 0.9

Input[0, 3] = 0.9
Input[1, 3] = 0.6

// ============================================
// Initial States
// ============================================
HiddenState[0, 0] = 0.0
HiddenState[1, 0] = 0.0
HiddenState[2, 0] = 0.0

CellState[0, 0] = 0.0
CellState[1, 0] = 0.0
CellState[2, 0] = 0.0

// ============================================
// LSTM Forward Pass
// ============================================

// Forget gate: decides what to forget from cell state
ForgetGate[i, *t+1] = sigmoid(
    W_f[i, j] HiddenState[j, *t]
  + U_f[i, k] Input[k, t]
)

// Input gate: decides what new information to store
InputGate[i, *t+1] = sigmoid(
    W_i[i, j] HiddenState[j, *t]
  + U_i[i, k] Input[k, t]
)

// Cell candidate: new candidate values
CellCandidate[i, *t+1] = tanh(
    W_c[i, j] HiddenState[j, *t]
  + U_c[i, k] Input[k, t]
)

// Update cell state: forget old + remember new
CellState[i, *t+1] = ForgetGate[i, *t+1] * CellState[i, *t]
                   + InputGate[i, *t+1] * CellCandidate[i, *t+1]

// Output gate: decides what to output
OutputGate[i, *t+1] = sigmoid(
    W_o[i, j] HiddenState[j, *t]
  + U_o[i, k] Input[k, t]
)

// Update hidden state
HiddenState[i, *t+1] = OutputGate[i, *t+1] * tanh(CellState[i, *t+1])

// ============================================
// Queries
// ============================================

// Query final states after processing all time steps
HiddenState[i, 0]?
CellState[i, 0]?