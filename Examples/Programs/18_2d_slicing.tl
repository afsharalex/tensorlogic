// ============================================
// 2D Tensor Slicing Example
// ============================================
// This program demonstrates slicing operations on 2D tensors (matrices)
// You can slice along rows, columns, or both dimensions simultaneously

// ============================================
// Define Input Matrix M (4 rows, 5 columns)
// ============================================
// Sample matrix with sequential values for easy verification
// Shape: [4, 5]
//
// Visual representation:
//         col0  col1  col2  col3  col4
// row 0 [  1.0   2.0   3.0   4.0   5.0 ]
// row 1 [  6.0   7.0   8.0   9.0  10.0 ]
// row 2 [ 11.0  12.0  13.0  14.0  15.0 ]
// row 3 [ 16.0  17.0  18.0  19.0  20.0 ]

// Initialize row 0
M[0, 0] = 1.0
M[0, 1] = 2.0
M[0, 2] = 3.0
M[0, 3] = 4.0
M[0, 4] = 5.0

// Initialize row 1
M[1, 0] = 6.0
M[1, 1] = 7.0
M[1, 2] = 8.0
M[1, 3] = 9.0
M[1, 4] = 10.0

// Initialize row 2
M[2, 0] = 11.0
M[2, 1] = 12.0
M[2, 2] = 13.0
M[2, 3] = 14.0
M[2, 4] = 15.0

// Initialize row 3
M[3, 0] = 16.0
M[3, 1] = 17.0
M[3, 2] = 18.0
M[3, 3] = 19.0
M[3, 4] = 20.0

// ============================================
// Operation 1: Extract Single Row
// ============================================
// Select row 1 (second row) using M[1, :]
// The colon means "all elements" in that dimension
//
// Result: [ 6.0  7.0  8.0  9.0  10.0 ]
// Shape: [5]

Row1 = M[1, :]

// ============================================
// Operation 2: Extract Single Column
// ============================================
// Select column 2 (third column) using M[:, 2]
// Gets all rows, but only column index 2
//
// Result: [ 3.0  8.0  13.0  18.0 ]
// Shape: [4]

Col2 = M[:, 2]

// ============================================
// Operation 3: Extract Row Slice
// ============================================
// Select rows 1 and 2 (indices 1, 2), all columns
// M[1:3, :] extracts a 2x5 sub-matrix
//
// Result:
// [  6.0   7.0   8.0   9.0  10.0 ]
// [ 11.0  12.0  13.0  14.0  15.0 ]
// Shape: [2, 5]

RowSlice = M[1:3, :]

// ============================================
// Operation 4: Extract Column Slice
// ============================================
// Select columns 1, 2, 3 (indices 1, 2, 3), all rows
// M[:, 1:4] extracts a 4x3 sub-matrix
//
// Result:
// [  2.0   3.0   4.0 ]
// [  7.0   8.0   9.0 ]
// [ 12.0  13.0  14.0 ]
// [ 17.0  18.0  19.0 ]
// Shape: [4, 3]

ColSlice = M[:, 1:4]

// ============================================
// Operation 5: Extract 2D Sub-Matrix
// ============================================
// Select both row and column slices simultaneously
// M[1:3, 2:5] extracts rows 1-2, columns 2-4
//
// Result:
// [  8.0   9.0  10.0 ]
// [ 13.0  14.0  15.0 ]
// Shape: [2, 3]

SubMatrix = M[1:3, 2:5]

// ============================================
// Operation 6: Extract Every Other Row
// ============================================
// Use step parameter to skip rows
// M[::2, :] extracts rows 0, 2 (every 2nd row)
//
// Result:
// [  1.0   2.0   3.0   4.0   5.0 ]
// [ 11.0  12.0  13.0  14.0  15.0 ]
// Shape: [2, 5]

EveryOtherRow = M[::2, :]

// ============================================
// Operation 7: Extract Every Other Column
// ============================================
// Skip columns using step parameter
// M[:, ::2] extracts columns 0, 2, 4
//
// Result:
// [  1.0   3.0   5.0 ]
// [  6.0   8.0  10.0 ]
// [ 11.0  13.0  15.0 ]
// [ 16.0  18.0  20.0 ]
// Shape: [4, 3]

EveryOtherCol = M[:, ::2]

// ============================================
// Operation 8: Extract Corner Sub-Matrix
// ============================================
// Extract top-left 2x2 corner
// M[:2, :2] gets rows 0-1, columns 0-1
//
// Result:
// [ 1.0  2.0 ]
// [ 6.0  7.0 ]
// Shape: [2, 2]

TopLeft = M[:2, :2]

// ============================================
// Operation 9: Extract Bottom-Right Corner
// ============================================
// Use negative indices to extract bottom-right 2x2 corner
// M[-2:, -2:] gets last 2 rows, last 2 columns
//
// Result:
// [ 14.0  15.0 ]
// [ 19.0  20.0 ]
// Shape: [2, 2]

BottomRight = M[-2:, -2:]

// ============================================
// Operation 10: Mixed Fixed and Slice Indices
// ============================================
// Fix one dimension to a specific index, slice the other
// M[2, 1:4] gets row 2, columns 1-3
//
// Result: [ 12.0  13.0  14.0 ]
// Shape: [3]

Row2Partial = M[2, 1:4]

// ============================================
// Query Results
// ============================================

// Query single row extraction
Row1[0]?  // Should return: 6.0
Row1[4]?  // Should return: 10.0

// Query single column extraction
Col2[0]?  // Should return: 3.0
Col2[3]?  // Should return: 18.0

// Query row slice
RowSlice[0, 0]?  // Should return: 6.0
RowSlice[1, 4]?  // Should return: 15.0

// Query column slice
ColSlice[0, 0]?  // Should return: 2.0
ColSlice[3, 2]?  // Should return: 19.0

// Query 2D sub-matrix
SubMatrix[0, 0]?  // Should return: 8.0
SubMatrix[1, 2]?  // Should return: 15.0

// Query stepped slices
EveryOtherRow[0, 0]?  // Should return: 1.0
EveryOtherRow[1, 4]?  // Should return: 15.0

EveryOtherCol[0, 0]?  // Should return: 1.0
EveryOtherCol[3, 2]?  // Should return: 20.0

// Query corner matrices
TopLeft[0, 0]?  // Should return: 1.0
TopLeft[1, 1]?  // Should return: 7.0

BottomRight[0, 0]?  // Should return: 14.0
BottomRight[1, 1]?  // Should return: 20.0

// Query mixed indexing
Row2Partial[0]?  // Should return: 12.0
Row2Partial[2]?  // Should return: 14.0
