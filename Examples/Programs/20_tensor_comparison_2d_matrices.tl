// 2D Matrix Comparisons
// Demonstrates element-wise comparisons on multi-dimensional tensors

// Create two 3x3 matrices for comparison
A[0, 0] = 1.0
A[0, 1] = 5.0
A[0, 2] = 3.0
A[1, 0] = 8.0
A[1, 1] = 2.0
A[1, 2] = 6.0
A[2, 0] = 4.0
A[2, 1] = 9.0
A[2, 2] = 7.0

B[0, 0] = 2.0
B[0, 1] = 4.0
B[0, 2] = 3.0
B[1, 0] = 7.0
B[1, 1] = 5.0
B[1, 2] = 6.0
B[2, 0] = 8.0
B[2, 1] = 1.0
B[2, 2] = 9.0

// Element-wise matrix comparisons
less_than[i, j] = A[i, j] < B[i, j]
greater_than[i, j] = A[i, j] > B[i, j]
equal[i, j] = A[i, j] == B[i, j]

// Threshold-based masking for matrices
threshold = 5.0
above_threshold[i, j] = A[i, j] > threshold
at_or_below[i, j] = A[i, j] <= threshold

// Apply mask: zero out values above threshold in A
masked_A[i, j] = A[i, j] * at_or_below[i, j]

// Range-based filtering for matrices
min_val = 3.0
max_val = 7.0
in_range[i, j] = (A[i, j] >= min_val) and (A[i, j] <= max_val)

// Extract only values in range
range_filtered[i, j] = A[i, j] * in_range[i, j]

// Comparison with scalar for entire matrix
scalar_threshold = 6.0
matrix_gt_scalar[i, j] = A[i, j] > scalar_threshold

// Combined condition: A > 5 AND B < 5
complex_condition[i, j] = (A[i, j] > 5.0) and (B[i, j] < 5.0)

// Application: attention masking (future positions)
// Create causal mask where position i can only attend to positions j <= i
// (For actual attention, you'd use this in an attention mechanism)
causal_mask[0, 0] = 1.0
causal_mask[0, 1] = 0.0
causal_mask[0, 2] = 0.0
causal_mask[1, 0] = 1.0
causal_mask[1, 1] = 1.0
causal_mask[1, 2] = 0.0
causal_mask[2, 0] = 1.0
causal_mask[2, 1] = 1.0
causal_mask[2, 2] = 1.0
