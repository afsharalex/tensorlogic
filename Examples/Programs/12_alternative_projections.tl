// ============================================
// Alternative Projection Operators Example
// ============================================
// This program demonstrates projection operators beyond summation:
// - max=  : Maximum pooling
// - avg=  : Average pooling
// - min=  : Minimum pooling
// - +=    : Sum (explicit, default)
//
// Projection operators determine how indices on the RHS
// that don't appear on the LHS are reduced/aggregated.

// ============================================
// Define Input Matrix X (4 rows × 3 columns)
// ============================================
// Sample data representing feature maps or activations
// Shape: [4, 3]
//
// Visual representation:
//        j=0   j=1   j=2
// i=0  [ 2.0   5.0   3.0 ]
// i=1  [ 1.0   8.0   4.0 ]
// i=2  [ 6.0   2.0   7.0 ]
// i=3  [ 3.0   9.0   1.0 ]

// Row 0
X[0, 0] = 2.0
X[0, 1] = 5.0
X[0, 2] = 3.0

// Row 1
X[1, 0] = 1.0
X[1, 1] = 8.0
X[1, 2] = 4.0

// Row 2
X[2, 0] = 6.0
X[2, 1] = 2.0
X[2, 2] = 7.0

// Row 3
X[3, 0] = 3.0
X[3, 1] = 9.0
X[3, 2] = 1.0

// ============================================
// Max Pooling
// ============================================
// Takes the maximum value over the j dimension
// For each row i, finds the largest value across all columns j
//
// Syntax: max= instead of = or +=
//
// Expected results:
// Y_max[0] = max(2.0, 5.0, 3.0) = 5.0
// Y_max[1] = max(1.0, 8.0, 4.0) = 8.0
// Y_max[2] = max(6.0, 2.0, 7.0) = 7.0
// Y_max[3] = max(3.0, 9.0, 1.0) = 9.0

Y_max[i] max= X[i, j]

// ============================================
// Average Pooling
// ============================================
// Takes the average (mean) value over the j dimension
// For each row i, computes the mean of all columns j
//
// Syntax: avg= instead of = or +=
//
// Expected results:
// Y_avg[0] = (2.0 + 5.0 + 3.0) / 3 = 10.0 / 3 ≈ 3.333
// Y_avg[1] = (1.0 + 8.0 + 4.0) / 3 = 13.0 / 3 ≈ 4.333
// Y_avg[2] = (6.0 + 2.0 + 7.0) / 3 = 15.0 / 3 = 5.0
// Y_avg[3] = (3.0 + 9.0 + 1.0) / 3 = 13.0 / 3 ≈ 4.333

Y_avg[i] avg= X[i, j]

// ============================================
// Min Pooling
// ============================================
// Takes the minimum value over the j dimension
// For each row i, finds the smallest value across all columns j
//
// Expected results:
// Y_min[0] = min(2.0, 5.0, 3.0) = 2.0
// Y_min[1] = min(1.0, 8.0, 4.0) = 1.0
// Y_min[2] = min(6.0, 2.0, 7.0) = 2.0
// Y_min[3] = min(3.0, 9.0, 1.0) = 1.0

Y_min[i] min= X[i, j]

// ============================================
// Sum Pooling (Explicit)
// ============================================
// Sums all values over the j dimension
// This is equivalent to default = operator
//
// Expected results:
// Y_sum[0] = 2.0 + 5.0 + 3.0 = 10.0
// Y_sum[1] = 1.0 + 8.0 + 4.0 = 13.0
// Y_sum[2] = 6.0 + 2.0 + 7.0 = 15.0
// Y_sum[3] = 3.0 + 9.0 + 1.0 = 13.0

Y_sum[i] += X[i, j]

// Equivalent to:
Y_sum_default[i] = X[i, j]

// ============================================
// Query Results
// ============================================

// Max pooling results
Y_max[0]?  // Should return: 5.0
Y_max[1]?  // Should return: 8.0
Y_max[2]?  // Should return: 7.0
Y_max[3]?  // Should return: 9.0

// Average pooling results
Y_avg[0]?  // Should return: ~3.333
Y_avg[1]?  // Should return: ~4.333
Y_avg[2]?  // Should return: 5.0
Y_avg[3]?  // Should return: ~4.333

// Min pooling results
Y_min[0]?  // Should return: 2.0
Y_min[1]?  // Should return: 1.0
Y_min[2]?  // Should return: 2.0
Y_min[3]?  // Should return: 1.0

// Sum pooling results
Y_sum[0]?  // Should return: 10.0
Y_sum[1]?  // Should return: 13.0
Y_sum[2]?  // Should return: 15.0
Y_sum[3]?  // Should return: 13.0

// Query all results at once
Y_max[i]?
Y_avg[i]?
Y_min[i]?
Y_sum[i]?