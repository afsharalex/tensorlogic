// ============================================
// Feature Engineering with Multiple Guards
// ============================================
// Create derived features based on multiple input conditions

// Input features
Age[0] = 5.0
Age[1] = 15.0
Age[2] = 25.0
Age[3] = 45.0
Age[4] = 70.0

Income[0] = 0.0
Income[1] = 20000.0
Income[2] = 50000.0
Income[3] = 80000.0
Income[4] = 40000.0

// Derived feature: Risk category
// Based on age and income combinations
// High values = higher risk

RiskScore[i] = 1.0 : (Age[i] < 18.0)
             | 0.5 : (Age[i] >= 18.0 and Age[i] < 30.0 and Income[i] < 30000.0)
             | 0.3 : (Age[i] >= 18.0 and Age[i] < 30.0 and Income[i] >= 30000.0)
             | 0.4 : (Age[i] >= 30.0 and Age[i] < 60.0 and Income[i] < 50000.0)
             | 0.2 : (Age[i] >= 30.0 and Age[i] < 60.0 and Income[i] >= 50000.0)
             | 0.6 : (Age[i] >= 60.0 and Income[i] < 40000.0)
             | 0.4 : (Age[i] >= 60.0)

// Expected results:
// RiskScore[0] = 1.0  (minor, high risk)
// RiskScore[1] = 0.5  (young adult, low income)
// RiskScore[2] = 0.2  (adult, good income)
// RiskScore[3] = 0.2  (adult, good income)
// RiskScore[4] = 0.6  (senior, low income)

RiskScore[i]?