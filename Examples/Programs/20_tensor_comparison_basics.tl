// Basic Tensor Comparison Examples
// Demonstrates element-wise boolean operations with comparison operators

// Example 1: Scalar comparisons
a = 5.0
b = 3.0
c = 5.0

// All comparison operators
lt = a < b        // Less than: 0.0 (false)
le1 = a <= b      // Less than or equal: 0.0 (false)
le2 = a <= c      // Less than or equal: 1.0 (true)
gt = a > b        // Greater than: 1.0 (true)
ge = a >= c       // Greater than or equal: 1.0 (true)
eq1 = a == b      // Equal: 0.0 (false)
eq2 = a == c      // Equal: 1.0 (true)
ne = a != b       // Not equal: 1.0 (true)

// Example 2: Element-wise vector comparisons
X[0] = 1.0
X[1] = 2.0
X[2] = 3.0
X[3] = 4.0
X[4] = 5.0

threshold = 3.0

// Create boolean masks
below[i] = X[i] < threshold       // [1, 1, 0, 0, 0]
above[i] = X[i] > threshold       // [0, 0, 0, 1, 1]
equal_to[i] = X[i] == threshold   // [0, 0, 1, 0, 0]
in_range[i] = X[i] >= threshold   // [0, 0, 1, 1, 1]

// Example 3: Using masks for filtering
// Zero out values below threshold
filtered[i] = X[i] * (X[i] >= threshold)

// Example 4: Tensor-to-tensor comparisons
A[0] = 1.0
A[1] = 3.0
A[2] = 5.0

B[0] = 2.0
B[1] = 3.0
B[2] = 4.0

// Element-wise comparisons between tensors
less[i] = A[i] < B[i]     // [1, 0, 0]
equal[i] = A[i] == B[i]   // [0, 1, 0]
greater[i] = A[i] > B[i]  // [0, 0, 1]
