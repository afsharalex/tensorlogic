// Advanced Tensor Comparison: Creating and Using Masks
// Demonstrates how comparison operators create boolean masks for data processing

// Dataset: Sensor readings with outliers and normal values
Readings[0] = 10.5
Readings[1] = 95.2   // Outlier (too high)
Readings[2] = 22.3
Readings[3] = -5.0   // Outlier (negative)
Readings[4] = 18.7
Readings[5] = 120.0  // Outlier (too high)
Readings[6] = 15.2
Readings[7] = 0.0    // Boundary case
Readings[8] = 25.0

// Define valid range
min_valid = 0.0
max_valid = 100.0

// Create boolean masks for different conditions
too_low[i] = Readings[i] < min_valid
too_high[i] = Readings[i] > max_valid
in_valid_range[i] = (Readings[i] >= min_valid) and (Readings[i] <= max_valid)

// Outlier detection: values outside [min_valid, max_valid]
is_outlier[i] = (Readings[i] < min_valid) or (Readings[i] > max_valid)

// Data cleaning: clip values to valid range
// If too low, set to min_valid; if too high, set to max_valid; otherwise keep original
clipped[i] = (min_valid * too_low[i]) +
             (max_valid * too_high[i]) +
             (Readings[i] * in_valid_range[i])

// Data filtering: zero out outliers, keep valid values
cleaned[i] = Readings[i] * in_valid_range[i]

// Categorical encoding: categorize readings into low/medium/high
low_threshold = 15.0
high_threshold = 20.0

is_low[i] = Readings[i] < low_threshold
is_medium[i] = (Readings[i] >= low_threshold) and (Readings[i] < high_threshold)
is_high[i] = Readings[i] >= high_threshold

// Count-based features: how many values fall into each category
// (These would need reduction in practice, shown here as per-element masks)
