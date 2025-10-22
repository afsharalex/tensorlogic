// ============================================
// Standard Normalization (Z-score)
// ============================================
// Transforms data to have mean=0 and standard deviation=1
// Formula: Y[i] = (X[i] - mean) / std_dev

// Data
X[0] = 2.0
X[1] = 4.0
X[2] = 6.0
X[3] = 8.0
X[4] = 10.0

// Compute mean
// mean = 30.0 / 5.0 = 6.0
mean = X[i] / 5.0

// Compute variance
// variance = sum((X[i] - mean)^2) / n
// First compute squared deviations
squared_dev[0] = -4.0
squared_dev[0] = squared_dev[0] * squared_dev[0]  // 16.0

squared_dev[1] = -2.0
squared_dev[1] = squared_dev[1] * squared_dev[1]  // 4.0

squared_dev[2] = 0.0
squared_dev[2] = squared_dev[2] * squared_dev[2]  // 0.0

squared_dev[3] = 2.0
squared_dev[3] = squared_dev[3] * squared_dev[3]  // 4.0

squared_dev[4] = 4.0
squared_dev[4] = squared_dev[4] * squared_dev[4]  // 16.0

// Sum of squared deviations = 40.0
// Variance = 40.0 / 5.0 = 8.0
variance = squared_dev[i] / 5.0

// Standard deviation = sqrt(8.0) ≈ 2.828
std_dev = sqrt(variance)

// Compute z-scores
// z[0] = (2.0 - 6.0) / 2.828 ≈ -1.414
// z[1] = (4.0 - 6.0) / 2.828 ≈ -0.707
// z[2] = (6.0 - 6.0) / 2.828 = 0.0
// z[3] = (8.0 - 6.0) / 2.828 ≈ 0.707
// z[4] = (10.0 - 6.0) / 2.828 ≈ 1.414
z_score[i] = (X[i] - mean) / std_dev

mean?       // 6.0
variance?   // 8.0
std_dev?    // ~2.828
z_score[2]? // 0.0