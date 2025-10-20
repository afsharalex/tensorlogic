// ============================================
// Arithmetic Operations Example
// ============================================
// This program demonstrates basic arithmetic operations:
// - Addition (e.g., adding bias terms)
// - Subtraction (e.g., mean centering)
// - Division (e.g., normalization)

// ============================================
// Define Input Vector X (5 elements)
// ============================================
// Sample data that we'll transform using arithmetic operations
// Shape: [5]
//
// Visual representation:
// i=0  i=1  i=2  i=3  i=4
// [ 2.0  4.0  6.0  8.0  10.0 ]

X[0] = 2.0
X[1] = 4.0
X[2] = 6.0
X[3] = 8.0
X[4] = 10.0

// ============================================
// Operation 1: Addition (Bias Term)
// ============================================
// Adding a bias vector to shift each element
// Common in neural networks: Y = X + b
//
// Define bias vector b (5 elements)
// Shape: [5]
//
// Visual representation:
// i=0  i=1  i=2  i=3  i=4
// [ 0.5  1.0  1.5  2.0  2.5 ]

b[0] = 0.5
b[1] = 1.0
b[2] = 1.5
b[3] = 2.0
b[4] = 2.5

// Compute addition
// Y1[0] = 2.0 + 0.5 = 2.5
// Y1[1] = 4.0 + 1.0 = 5.0
// Y1[2] = 6.0 + 1.5 = 7.5
// Y1[3] = 8.0 + 2.0 = 10.0
// Y1[4] = 10.0 + 2.5 = 12.5

Y1[i] = X[i] + b[i]

// ============================================
// Operation 2: Subtraction (Mean Centering)
// ============================================
// Subtracting the mean to center data around zero
// Common preprocessing step in machine learning
//
// First compute the mean of X
// mean = (X[0] + X[1] + X[2] + X[3] + X[4]) / 5
//      = (2.0 + 4.0 + 6.0 + 8.0 + 10.0) / 5
//      = 30.0 / 5
//      = 6.0

mean = X[i] / 5.0

// Compute mean-centered values
// Y2[0] = 2.0 - 6.0 = -4.0
// Y2[1] = 4.0 - 6.0 = -2.0
// Y2[2] = 6.0 - 6.0 = 0.0
// Y2[3] = 8.0 - 6.0 = 2.0
// Y2[4] = 10.0 - 6.0 = 4.0
//
// Note: Mean of Y2 will be zero (centered distribution)

Y2[i] = X[i] - mean

// ============================================
// Operation 3: Division (Normalization)
// ============================================
// Dividing by a constant to normalize/scale values
// Common for converting to unit scale or probability-like values
//
// Define normalization constant Z
// Z represents the sum or scale factor
// Z = sum of X = 30.0

Z = X[i]

// Compute normalized values (divide each element by sum)
// Y3[0] = 2.0 / 30.0 ≈ 0.0667 (6.67%)
// Y3[1] = 4.0 / 30.0 ≈ 0.1333 (13.33%)
// Y3[2] = 6.0 / 30.0 = 0.2000 (20.00%)
// Y3[3] = 8.0 / 30.0 ≈ 0.2667 (26.67%)
// Y3[4] = 10.0 / 30.0 ≈ 0.3333 (33.33%)
//
// Note: Sum of Y3 will equal 1.0 (normalized distribution)

Y3[i] = X[i] / Z

// ============================================
// Query Results
// ============================================

// Query addition results
Y1[0]?  // Should return: 2.5
Y1[2]?  // Should return: 7.5
Y1[4]?  // Should return: 12.5

// Query subtraction results (mean-centered)
mean?   // Should return: 6.0
Y2[0]?  // Should return: -4.0
Y2[2]?  // Should return: 0.0
Y2[4]?  // Should return: 4.0

// Query division results (normalized)
Z?      // Should return: 30.0
Y3[0]?  // Should return: ~0.0667
Y3[2]?  // Should return: 0.2
Y3[4]?  // Should return: ~0.3333

// Verify normalized sum equals 1.0
normalized_sum = Y3[i]
normalized_sum?  // Should return: 1.0