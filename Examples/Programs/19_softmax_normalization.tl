// ============================================
// Softmax Normalization with Normalized Indices
// ============================================
// This program demonstrates the core normalized index feature:
// converting raw scores into probability distributions
//
// Normalized indices (i.) indicate which dimension should be normalized
// such that the values along that dimension sum to 1.0
//
// Paper reference (Domingos, 2025):
//   "The notation p'. indicates that p' is the index to be normalized"

// ============================================
// Example 1: Classification Logits
// ============================================
// Raw logits for 3 classes, 2 examples
// Shape: [examples, classes] = [2, 3]

Logits[0, 0] = 2.0   // Example 0: Class 0 score
Logits[0, 1] = 1.0   // Example 0: Class 1 score
Logits[0, 2] = 0.1   // Example 0: Class 2 score

Logits[1, 0] = 0.5   // Example 1: Class 0 score
Logits[1, 1] = 2.5   // Example 1: Class 1 score
Logits[1, 2] = 1.5   // Example 1: Class 2 score

// Convert logits to probabilities using normalized indices
// The "c." notation means: normalize over the c (class) dimension
// For each example e, the probabilities over all classes will sum to 1.0
//
// This is equivalent to:
//   Probs[e, c] = exp(Logits[e, c]) / Σ_c' exp(Logits[e, c'])
//
// But much more concise!

Probs[e, c.] = softmax(Logits[e, c])

// Query the probability distributions
Probs[0, 0]?  // P(Class 0 | Example 0)
Probs[0, 1]?  // P(Class 1 | Example 0)
Probs[0, 2]?  // P(Class 2 | Example 0)

// Verify normalization: should sum to 1.0
prob_sum_0 = Probs[0, c]
prob_sum_0?  // Should be 1.0

// Check second example
Probs[1, 0]?
Probs[1, 1]?
Probs[1, 2]?

prob_sum_1 = Probs[1, c]
prob_sum_1?  // Should be 1.0

// ============================================
// Example 2: Comparing Normalization Dimensions
// ============================================
// Matrix of compatibility scores
// Shape: [queries, keys] = [2, 3]

Scores[0, 0] = 0.9
Scores[0, 1] = 0.5
Scores[0, 2] = 0.3

Scores[1, 0] = 0.6
Scores[1, 1] = 0.8
Scores[1, 2] = 0.4

// Normalize over keys (for each query, normalize across keys)
// This is what attention mechanisms do!
// For each query q, the weights over all keys sum to 1.0
Attn_over_keys[q, k.] = softmax(Scores[q, k])

// Normalize over queries (for each key, normalize across queries)
// Less common, but sometimes useful
// For each key k, the weights over all queries sum to 1.0
Attn_over_queries[q., k] = softmax(Scores[q, k])

// Query both normalizations
Attn_over_keys[0, 0]?
Attn_over_keys[0, 1]?
Attn_over_keys[0, 2]?

Attn_over_queries[0, 0]?
Attn_over_queries[1, 0]?

// Verify different normalization directions
sum_over_keys = Attn_over_keys[0, k]
sum_over_keys?  // Should be 1.0

sum_over_queries = Attn_over_queries[q, 0]
sum_over_queries?  // Should be 1.0

// ============================================
// Example 3: Multi-dimensional Normalization
// ============================================
// 3D tensor: [batch, sequence, features]
// Shape: [2, 3, 4]

// Initialize a 3D tensor with some values
Features3D[0, 0, 0] = 1.0
Features3D[0, 0, 1] = 2.0
Features3D[0, 0, 2] = 3.0
Features3D[0, 0, 3] = 4.0

Features3D[0, 1, 0] = 0.5
Features3D[0, 1, 1] = 1.5
Features3D[0, 1, 2] = 2.5
Features3D[0, 1, 3] = 3.5

// Normalize over the feature dimension (f) for each (batch, sequence) pair
// For each combination of b and s, the features sum to 1.0
Normalized[b, s, f.] = softmax(Features3D[b, s, f])

// Query normalized features
Normalized[0, 0, 0]?
Normalized[0, 0, 1]?
Normalized[0, 0, 2]?
Normalized[0, 0, 3]?

// Verify: features for batch 0, sequence 0 sum to 1.0
feature_sum = Normalized[0, 0, f]
feature_sum?  // Should be 1.0

// ============================================
// Key Semantic Notes
// ============================================
// 1. The normalized index (i.) MUST appear on the LHS
//    - Correct: Y[i, j.] = softmax(X[i, j])
//    - The dot indicates normalization over j for each i
//
// 2. Only ONE index can be normalized per equation
//    - Correct: Y[i, j.] = softmax(X[i, j])
//    - Invalid: Y[i., j.] = softmax(X[i, j])  // Can't normalize both
//
// 3. The normalized index creates a probability distribution
//    - For each setting of the non-normalized indices
//    - The values along the normalized dimension sum to 1.0
//
// 4. softmax is the typical normalization function
//    - But other normalizations could be supported (L1, L2, etc.)
