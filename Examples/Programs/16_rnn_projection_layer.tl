// ============================================
// Recurrent Neural Network with Input Projection Layer
// ============================================
// This program implements an RNN that:
// 1. Projects lower-dimensional input to hidden dimension
// 2. Processes sequence using virtual time index (*t)
// 3. Produces final output prediction
//
// Architecture:
// - Input dimension: 3
// - Hidden dimension: 4
// - Sequence length: 5
// - Output dimension: 1 (binary classification)

// ============================================
// Network Parameters
// ============================================

// Recurrent weights (hidden-to-hidden)
// Shape: [4, 4]
// Maps previous hidden state to current hidden state
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

// Input projection weights (input-to-hidden)
// Shape: [4, 3]
// Projects 3D input to 4D hidden space
U_proj[0, 0] = 0.7
U_proj[0, 1] = 0.3
U_proj[0, 2] = 0.2
U_proj[1, 0] = 0.4
U_proj[1, 1] = 0.6
U_proj[1, 2] = 0.1
U_proj[2, 0] = 0.5
U_proj[2, 1] = 0.2
U_proj[2, 2] = 0.8
U_proj[3, 0] = 0.3
U_proj[3, 1] = 0.5
U_proj[3, 2] = 0.4

// Bias for hidden layer
// Shape: [4]
b[0] = 0.1
b[1] = 0.2
b[2] = 0.1
b[3] = 0.3

// ============================================
// Input Sequence
// ============================================
// Shape: [3, 5] (3 input features, 5 time steps)
// Example: Sensor readings over time
// Feature 0: Temperature
// Feature 1: Pressure
// Feature 2: Humidity

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
// Shape: [4]
// Initialize to zeros
State[0, 0] = 0.0
State[1, 0] = 0.0
State[2, 0] = 0.0
State[3, 0] = 0.0

// ============================================
// Step 1: Project Input to Hidden Dimension
// ============================================
// Transform 3D input features to 4D hidden representation
// Input_proj[i, t] = Σ_k U_proj[i, k] * Input[k, t]
//
// For each time step t and each hidden unit i:
// This computes a weighted combination of the 3 input features
//
// Shape: [4, 5] (4 hidden units, 5 time steps)
//
// Example calculation for Input_proj[0, 0]:
// Input_proj[0, 0] = U_proj[0,0]*Input[0,0] + U_proj[0,1]*Input[1,0] + U_proj[0,2]*Input[2,0]
//                  = 0.7*1.0 + 0.3*0.5 + 0.2*0.8
//                  = 0.7 + 0.15 + 0.16
//                  = 1.01

Input_proj[i, t] = U_proj[i, k] Input[k, t]

// ============================================
// Step 2: RNN Recurrent Update
// ============================================
// Process sequence using virtual time index
// Virtual index *t means successive values overwrite same memory
//
// Update equation:
// State[i, *t+1] = relu(W[i,j] * State[j,*t] + Input_proj[i,t] + b[i])
//
// Components:
// 1. W[i,j] * State[j,*t]: Recurrent connection from previous state
//    - Sums over j (previous hidden units)
//    - Produces vector of length i (current hidden units)
//
// 2. Input_proj[i,t]: Current time step's projected input
//    - Already in correct dimension [4]
//    - No summation needed
//
// 3. b[i]: Bias term
//
// 4. relu(): Non-linearity
//
// Execution per time step:
// t=0: State[:,*1] = relu(W @ State[:,0] + Input_proj[:,0] + b)
// t=1: State[:,*2] = relu(W @ State[:,*1] + Input_proj[:,1] + b)
// t=2: State[:,*3] = relu(W @ State[:,*2] + Input_proj[:,2] + b)
// t=3: State[:,*4] = relu(W @ State[:,*3] + Input_proj[:,3] + b)
// t=4: State[:,*5] = relu(W @ State[:,*4] + Input_proj[:,4] + b)

State[i, *t+1] = relu(
    W[i, j] State[j, *t]
  + Input_proj[i, t]
  + b[i]
)

// ============================================
// Step 3: Output Layer
// ============================================
// Map final hidden state to output prediction
// Shape: W_out[4], State[4, *5] → Output (scalar)

W_out[0] = 0.6
W_out[1] = 0.4
W_out[2] = 0.5
W_out[3] = 0.3

bias_out = -0.2

// Compute output from final hidden state
// Output = sigmoid(W_out · State[:, *5] + bias_out)
//
// This produces a value between 0 and 1
// Can be interpreted as probability for binary classification
Output = sigmoid(W_out[i] State[i, *5] + bias_out)

// ============================================
// Queries: Inspect Intermediate Computations
// ============================================

// Query projected inputs at different time steps
Input_proj[0, 0]?  // First hidden unit, first time step
Input_proj[1, 2]?  // Second hidden unit, third time step
Input_proj[i, 0]?  // All hidden units at first time step

// Query final hidden state after all time steps
State[0, *5]?  // First hidden unit's final value
State[1, *5]?  // Second hidden unit's final value
State[2, *5]?  // Third hidden unit's final value
State[3, *5]?  // Fourth hidden unit's final value
State[i, *5]?  // All hidden units' final values

// Query final output
Output?  // Final prediction (probability)