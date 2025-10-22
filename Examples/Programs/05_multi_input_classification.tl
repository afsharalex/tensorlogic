// ============================================
// Perceptron for Multi-Feature Classification
// ============================================
// Classifying whether a student passes based on:
// - Hours studied (X[0])
// - Previous test score (X[1])
// - Attendance rate (X[2])

// Learned weights
W[0] = 0.4   // Hours studied importance
W[1] = 0.3   // Previous score importance
W[2] = 0.2   // Attendance importance

// Bias term
bias = -0.6

// Student 1: Likely to pass
// 8 hours studied, 85% previous score, 90% attendance
Student1[0] = 8.0
Student1[1] = 85.0
Student1[2] = 90.0

Pass1 = step(W[i] Student1[i] + bias)
// Calculation: 0.4*8 + 0.3*85 + 0.2*90 - 0.6
//            = 3.2 + 25.5 + 18.0 - 0.6
//            = 46.1 -> step(46.1) = 1 (PASS)

// Student 2: Likely to fail
// 2 hours studied, 50% previous score, 60% attendance
Student2[0] = 2.0
Student2[1] = 50.0
Student2[2] = 60.0

Pass2 = step(W[i] Student2[i] + bias)
// Calculation: 0.4*2 + 0.3*50 + 0.2*60 - 0.6
//            = 0.8 + 15.0 + 12.0 - 0.6
//            = 27.2 -> step(27.2) = 1 (PASS - but borderline)

Pass1?  // Should return: 1 (pass)
Pass2?  // Should return: 1 (pass)