// ============================================
// Slice Assignment Example
// ============================================
// This program demonstrates how to modify tensors using slice assignment
// Slice assignment allows in-place updates to specific regions of a tensor
// without creating copies - useful for efficient tensor manipulation

// ============================================
// Define Base Vector V (10 elements)
// ============================================
// Initialize with zeros, then selectively update slices
// Shape: [10]
//
// Initial state:
// i=0  i=1  i=2  i=3  i=4  i=5  i=6  i=7  i=8  i=9
// [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ]

V[0] = 0.0
V[1] = 0.0
V[2] = 0.0
V[3] = 0.0
V[4] = 0.0
V[5] = 0.0
V[6] = 0.0
V[7] = 0.0
V[8] = 0.0
V[9] = 0.0

// ============================================
// Operation 1: Update Middle Section
// ============================================
// Replace elements at indices 3, 4, 5 with [10, 20, 30]
// V[3:6] = [10, 20, 30]
//
// After assignment:
// [ 0.0  0.0  0.0  10.0  20.0  30.0  0.0  0.0  0.0  0.0 ]

V[3:6] = [10.0, 20.0, 30.0]

// ============================================
// Operation 2: Update Beginning
// ============================================
// Replace first 2 elements with [1, 2]
// V[:2] = [1, 2]
//
// After assignment:
// [ 1.0  2.0  0.0  10.0  20.0  30.0  0.0  0.0  0.0  0.0 ]

V[:2] = [1.0, 2.0]

// ============================================
// Operation 3: Update End
// ============================================
// Replace last 3 elements with [7, 8, 9]
// V[-3:] = [7, 8, 9]
//
// After assignment:
// [ 1.0  2.0  0.0  10.0  20.0  30.0  0.0  7.0  8.0  9.0 ]

V[-3:] = [7.0, 8.0, 9.0]

// ============================================
// Query updated vector
// ============================================
V[0]?   // Should return: 1.0
V[1]?   // Should return: 2.0
V[3]?   // Should return: 10.0
V[4]?   // Should return: 20.0
V[5]?   // Should return: 30.0
V[7]?   // Should return: 7.0
V[9]?   // Should return: 9.0

// ============================================
// 2D Slice Assignment Example
// ============================================
// Create a 4x4 matrix and update regions

// Define base matrix M (4x4)
// Initial state:
//      col0  col1  col2  col3
// row0 [ 0.0   0.0   0.0   0.0 ]
// row1 [ 0.0   0.0   0.0   0.0 ]
// row2 [ 0.0   0.0   0.0   0.0 ]
// row3 [ 0.0   0.0   0.0   0.0 ]

M[0, 0] = 0.0
M[0, 1] = 0.0
M[0, 2] = 0.0
M[0, 3] = 0.0
M[1, 0] = 0.0
M[1, 1] = 0.0
M[1, 2] = 0.0
M[1, 3] = 0.0
M[2, 0] = 0.0
M[2, 1] = 0.0
M[2, 2] = 0.0
M[2, 3] = 0.0
M[3, 0] = 0.0
M[3, 1] = 0.0
M[3, 2] = 0.0
M[3, 3] = 0.0

// ============================================
// Update Top Row
// ============================================
// Set first row to [1, 2, 3, 4]
// M[0, :] = [1, 2, 3, 4]
//
// After assignment:
//      col0  col1  col2  col3
// row0 [ 1.0   2.0   3.0   4.0 ]
// row1 [ 0.0   0.0   0.0   0.0 ]
// row2 [ 0.0   0.0   0.0   0.0 ]
// row3 [ 0.0   0.0   0.0   0.0 ]

M[0, :] = [1.0, 2.0, 3.0, 4.0]

// ============================================
// Update Center 2x2 Sub-Matrix
// ============================================
// Update the center region (rows 1-2, cols 1-2)
// This assigns a 2x2 matrix to a 2x2 slice
//
// After assignment:
//      col0  col1  col2  col3
// row0 [ 1.0   2.0   3.0   4.0 ]
// row1 [ 0.0   5.0   6.0   0.0 ]
// row2 [ 0.0   7.0   8.0   0.0 ]
// row3 [ 0.0   0.0   0.0   0.0 ]

M[1:3, 1:3] = [[5.0, 6.0], [7.0, 8.0]]

// ============================================
// Update Last Column
// ============================================
// Set the last column to [10, 20, 30, 40]
// M[:, 3] = [10, 20, 30, 40]
//
// After assignment:
//      col0  col1  col2  col3
// row0 [ 1.0   2.0   3.0  10.0 ]
// row1 [ 0.0   5.0   6.0  20.0 ]
// row2 [ 0.0   7.0   8.0  30.0 ]
// row3 [ 0.0   0.0   0.0  40.0 ]

M[:, 3] = [10.0, 20.0, 30.0, 40.0]

// ============================================
// Query Updated Matrix
// ============================================
M[0, 0]?  // Should return: 1.0
M[0, 3]?  // Should return: 10.0
M[1, 1]?  // Should return: 5.0
M[2, 2]?  // Should return: 8.0
M[2, 3]?  // Should return: 30.0
M[3, 3]?  // Should return: 40.0

// ============================================
// Practical Example: Batch Data Preparation
// ============================================
// Common use case in neural networks: initialize batch with padding,
// then fill with actual data

// Create batch of 3 sequences, each of length 5
// Initialize with padding value -1.0
Batch[0, 0] = -1.0
Batch[0, 1] = -1.0
Batch[0, 2] = -1.0
Batch[0, 3] = -1.0
Batch[0, 4] = -1.0
Batch[1, 0] = -1.0
Batch[1, 1] = -1.0
Batch[1, 2] = -1.0
Batch[1, 3] = -1.0
Batch[1, 4] = -1.0
Batch[2, 0] = -1.0
Batch[2, 1] = -1.0
Batch[2, 2] = -1.0
Batch[2, 3] = -1.0
Batch[2, 4] = -1.0

// Fill sequence 1 (length 3)
Batch[0, :3] = [1.0, 2.0, 3.0]

// Fill sequence 2 (length 5, full)
Batch[1, :] = [4.0, 5.0, 6.0, 7.0, 8.0]

// Fill sequence 3 (length 2)
Batch[2, :2] = [9.0, 10.0]

// Result:
//       pos0  pos1  pos2  pos3  pos4
// seq0 [ 1.0   2.0   3.0  -1.0  -1.0 ]  (padded)
// seq1 [ 4.0   5.0   6.0   7.0   8.0 ]  (full)
// seq2 [ 9.0  10.0  -1.0  -1.0  -1.0 ]  (padded)

// ============================================
// Query Batch Results
// ============================================
Batch[0, 0]?  // Should return: 1.0
Batch[0, 3]?  // Should return: -1.0 (padding)
Batch[1, 4]?  // Should return: 8.0
Batch[2, 1]?  // Should return: 10.0
Batch[2, 2]?  // Should return: -1.0 (padding)
