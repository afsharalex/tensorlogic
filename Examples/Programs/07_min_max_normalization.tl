// ============================================
// Min-Max Normalization
// ============================================
// Scales data to [0, 1] range
// Formula: Y[i] = (X[i] - min) / (max - min)

// Data with varying range
X[0] = 10.0
X[1] = 25.0
X[2] = 15.0
X[3] = 40.0
X[4] = 20.0

// Find minimum (manually for this example)
min_val = 10.0

// Find maximum (manually for this example)
max_val = 40.0

// Compute range
range_val = max_val - min_val  // 30.0

// Normalize to [0, 1]
// Y[0] = (10.0 - 10.0) / 30.0 = 0.0
// Y[1] = (25.0 - 10.0) / 30.0 = 0.5
// Y[2] = (15.0 - 10.0) / 30.0 ≈ 0.167
// Y[3] = (40.0 - 10.0) / 30.0 = 1.0
// Y[4] = (20.0 - 10.0) / 30.0 ≈ 0.333
Y_normalized[i] = (X[i] - min_val) / range_val

Y_normalized[0]?  // 0.0 (minimum)
Y_normalized[3]?  // 1.0 (maximum)
Y_normalized[1]?  // 0.5 (middle)