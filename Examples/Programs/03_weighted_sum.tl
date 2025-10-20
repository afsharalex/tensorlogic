// ============================================
// Inner Product as Weighted Sum
// ============================================
// Common use case: combining features with learned weights

// Features extracted from an input
features = [0.8, 0.3, 0.9, 0.1, 0.7]

// Learned weights for each feature
weights = [2.5, 1.0, 3.0, 0.5, 1.5]

// Compute weighted sum (commonly used in linear models)
prediction = weights[i] features[i]

// Step-by-step:
// 2.5*0.8 + 1.0*0.3 + 3.0*0.9 + 0.5*0.1 + 1.5*0.7
// = 2.0 + 0.3 + 2.7 + 0.05 + 1.05
// = 6.1

prediction?  // Expected: 6.1