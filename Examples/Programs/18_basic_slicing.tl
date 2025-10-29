// ============================================
// Basic Slicing Operations Example
// ============================================
// This program demonstrates fundamental slicing operations on 1D tensors
// using Python-style slice syntax: [start:end:step]
//
// Slice notation: [start:end:step]
// - start: first index to include (inclusive)
// - end: last index to include (exclusive)
// - step: increment between indices (optional, default=1)

// ============================================
// Define Input Vector X (10 elements)
// ============================================
// Sample sequence of numbers from 0 to 9
// Shape: [10]
//
// Visual representation:
// i=0  i=1  i=2  i=3  i=4  i=5  i=6  i=7  i=8  i=9
// [ 0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 ]

X[0] = 0.0
X[1] = 1.0
X[2] = 2.0
X[3] = 3.0
X[4] = 4.0
X[5] = 5.0
X[6] = 6.0
X[7] = 7.0
X[8] = 8.0
X[9] = 9.0

// ============================================
// Operation 1: Middle Slice [start:end]
// ============================================
// Extract elements from index 2 to 7 (exclusive)
// Y1 = X[2:7] extracts indices 2, 3, 4, 5, 6
//
// Result: [ 2.0  3.0  4.0  5.0  6.0 ]
// Shape: [5]

Y1 = X[2:7]

// ============================================
// Operation 2: Slice from Beginning [:end]
// ============================================
// Extract first 4 elements (indices 0, 1, 2, 3)
// Y2 = X[:4] is equivalent to X[0:4]
//
// Result: [ 0.0  1.0  2.0  3.0 ]
// Shape: [4]

Y2 = X[:4]

// ============================================
// Operation 3: Slice to End [start:]
// ============================================
// Extract elements from index 6 to the end
// Y3 = X[6:] extracts indices 6, 7, 8, 9
//
// Result: [ 6.0  7.0  8.0  9.0 ]
// Shape: [4]

Y3 = X[6:]

// ============================================
// Operation 4: Full Slice [:]
// ============================================
// Extract all elements (creates a copy)
// Y4 = X[:] is equivalent to X[0:10]
//
// Result: [ 0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 ]
// Shape: [10]

Y4 = X[:]

// ============================================
// Operation 5: Slice with Step [start:end:step]
// ============================================
// Extract every 2nd element from index 0 to 8
// Y5 = X[0:8:2] extracts indices 0, 2, 4, 6
//
// Result: [ 0.0  2.0  4.0  6.0 ]
// Shape: [4]

Y5 = X[0:8:2]

// ============================================
// Operation 6: Slice with Step from Beginning [::step]
// ============================================
// Extract every 3rd element from the entire array
// Y6 = X[::3] extracts indices 0, 3, 6, 9
//
// Result: [ 0.0  3.0  6.0  9.0 ]
// Shape: [4]

Y6 = X[::3]

// ============================================
// Operation 7: Negative Indices
// ============================================
// Use negative indices to count from the end
// X[-3:] extracts the last 3 elements (indices 7, 8, 9)
//
// Result: [ 7.0  8.0  9.0 ]
// Shape: [3]

Y7 = X[-3:]

// ============================================
// Operation 8: Negative Start and End
// ============================================
// Extract middle section using negative indexing
// X[-7:-2] extracts from index 3 to index 7 (indices 3, 4, 5, 6, 7)
//
// Result: [ 3.0  4.0  5.0  6.0  7.0 ]
// Shape: [5]

Y8 = X[-7:-2]

// ============================================
// Query Results
// ============================================

// Query middle slice results
Y1[0]?  // Should return: 2.0 (first element of slice)
Y1[2]?  // Should return: 4.0 (middle element of slice)
Y1[4]?  // Should return: 6.0 (last element of slice)

// Query beginning slice
Y2[0]?  // Should return: 0.0
Y2[3]?  // Should return: 3.0

// Query end slice
Y3[0]?  // Should return: 6.0
Y3[3]?  // Should return: 9.0

// Query full slice
Y4[0]?  // Should return: 0.0
Y4[9]?  // Should return: 9.0

// Query step slices
Y5[0]?  // Should return: 0.0
Y5[1]?  // Should return: 2.0
Y5[3]?  // Should return: 6.0

Y6[0]?  // Should return: 0.0
Y6[1]?  // Should return: 3.0
Y6[3]?  // Should return: 9.0

// Query negative index slices
Y7[0]?  // Should return: 7.0
Y7[2]?  // Should return: 9.0

Y8[0]?  // Should return: 3.0
Y8[4]?  // Should return: 7.0
