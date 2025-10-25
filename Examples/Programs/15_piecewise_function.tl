// ============================================
// Piecewise Function Definition
// ============================================
// Define a mathematical piecewise function using guards

// Input values
X[0] = -5.0
X[1] = -3.0
X[2] = -1.0
X[3] = 0.0
X[4] = 1.0
X[5] = 3.0
X[6] = 5.0
X[7] = 7.0

// Piecewise function:
// f(x) = x^2        if x < 0
//      = 0          if x == 0
//      = sqrt(x)    if 0 < x <= 4
//      = 2*x        if x > 4

Y[i] = X[i] * X[i] : (X[i] < 0.0)
     | 0.0 : (X[i] == 0.0)
     | sqrt(X[i]) : (X[i] > 0.0 and X[i] <= 4.0)
     | 2.0 * X[i]

// Expected results:
// Y[0] = (-5)^2 = 25.0       (x < 0, first guard)
// Y[1] = (-3)^2 = 9.0        (x < 0, first guard)
// Y[2] = (-1)^2 = 1.0        (x < 0, first guard)
// Y[3] = 0.0                 (x == 0, second guard)
// Y[4] = sqrt(1) = 1.0       (0 < x <= 4, third guard)
// Y[5] = sqrt(3) ≈ 1.732     (0 < x <= 4, third guard)
// Y[6] = 2*5 = 10.0          (x > 4, default guard)
// Y[7] = 2*7 = 14.0          (x > 4, default guard)

Y[0]?  // 25.0
Y[3]?  // 0.0
Y[5]?  // ~1.732
Y[7]?  // 14.0