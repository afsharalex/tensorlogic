// ============================================
// Guarded Clause Example: Conditional Weighting
// ============================================
// This program demonstrates the new guarded clause syntax
// for applying different transformations based on conditions
//
// Use case: Apply different weights to data based on index ranges
// - High weight (1.0) for indices outside middle range
// - Medium weight (0.5) for even indices in middle range
// - Low weight (0.1) for odd indices in middle range (default)

// ============================================
// Input Data
// ============================================
// Define a dataset with 30 elements
// Shape: [30]

X[0] = 1.0
X[1] = 2.0
X[2] = 3.0
X[3] = 4.0
X[4] = 5.0
X[5] = 6.0
X[6] = 7.0
X[7] = 8.0
X[8] = 9.0
X[9] = 10.0
X[10] = 11.0
X[11] = 12.0
X[12] = 13.0
X[13] = 14.0
X[14] = 15.0
X[15] = 16.0
X[16] = 17.0
X[17] = 18.0
X[18] = 19.0
X[19] = 20.0
X[20] = 21.0
X[21] = 22.0
X[22] = 23.0
X[23] = 24.0
X[24] = 25.0
X[25] = 26.0
X[26] = 27.0
X[27] = 28.0
X[28] = 29.0
X[29] = 30.0

// ============================================
// Guarded Clause: Conditional Weighting
// ============================================
// Apply different weights based on index value
//
// Guard syntax: expression : condition
// Multiple guards are tried in order (like pattern matching)
// The first matching guard is used
//
// Logic:
// - If i < 10 OR i > 20: apply weight 1.0 (high priority)
// - Else if i >= 10 AND i <= 20 AND i is even: apply weight 0.5
// - Else (default): apply weight 0.1

Weighted[i] = 1.0 * X[i] : (i < 10)
            | 1.0 * X[i] : (i > 20)
            | 0.5 * X[i] : (i >= 10 and i <= 20 and i % 2 == 0)
            | 0.1 * X[i]

// Expected results:
// Weighted[0] = 1.0 * 1.0 = 1.0    (i=0 < 10, first guard matches)
// Weighted[1] = 1.0 * 2.0 = 2.0    (i=1 < 10, first guard matches)
// ...
// Weighted[9] = 1.0 * 10.0 = 10.0  (i=9 < 10, first guard matches)
// Weighted[10] = 0.5 * 11.0 = 5.5  (i=10, even, third guard matches)
// Weighted[11] = 0.1 * 12.0 = 1.2  (i=11, odd, default matches)
// Weighted[12] = 0.5 * 13.0 = 6.5  (i=12, even, third guard matches)
// ...
// Weighted[20] = 0.5 * 21.0 = 10.5 (i=20, even, third guard matches)
// Weighted[21] = 1.0 * 22.0 = 22.0 (i=21 > 20, second guard matches)
// ...
// Weighted[29] = 1.0 * 30.0 = 30.0 (i=29 > 20, second guard matches)

// ============================================
// Queries
// ============================================

// Query specific weighted values
Weighted[0]?   // Should return: 1.0 (high weight, i < 10)
Weighted[5]?   // Should return: 6.0 (high weight, i < 10)
Weighted[10]?  // Should return: 5.5 (medium weight, even in middle)
Weighted[11]?  // Should return: 1.2 (low weight, odd in middle)
Weighted[12]?  // Should return: 6.5 (medium weight, even in middle)
Weighted[15]?  // Should return: 1.6 (low weight, odd in middle)
Weighted[20]?  // Should return: 10.5 (medium weight, even in middle)
Weighted[25]?  // Should return: 26.0 (high weight, i > 20)

// Query all weighted values
Weighted[i]?