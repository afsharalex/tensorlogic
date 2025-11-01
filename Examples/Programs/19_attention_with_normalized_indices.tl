// ============================================
// Attention Mechanism with Normalized Indices
// ============================================
// This program demonstrates the normalized index feature (i.) from the
// Tensor Logic paper (Domingos, 2025, Section 4.1, Table 2)
//
// The dot notation "p'." indicates that softmax normalization should be
// applied over the p' dimension for each value of p.
//
// Key concept:
// - Regular index: p_prime (sums over all values)
// - Normalized index: p_prime. (applies softmax, creating probability distribution)
//
// Paper reference:
//   "The notation p'. indicates that p' is the index to be normalized
//    (i.e., for each p, softmax is applied to the vector indexed by p')."

// ============================================
// Input: Sequence Embeddings
// ============================================
// Simple 3-token sequence with 2-dimensional embeddings
// Representing: "The cat sat"

X[0, 0] = 1.0
X[0, 1] = 0.5

X[1, 0] = 0.6
X[1, 1] = 1.2

X[2, 0] = 0.9
X[2, 1] = 0.7

// ============================================
// Attention Parameters
// ============================================
// Simplified weights for clarity
// d_k = 2 (query/key dimension)

WQ[0, 0] = 0.5
WQ[0, 1] = 0.3
WQ[1, 0] = 0.4
WQ[1, 1] = 0.6

WK[0, 0] = 0.6
WK[0, 1] = 0.2
WK[1, 0] = 0.3
WK[1, 1] = 0.5

WV[0, 0] = 0.7
WV[0, 1] = 0.4
WV[1, 0] = 0.2
WV[1, 1] = 0.8

// Scaling factor
sqrt_dk = 1.414

// ============================================
// Attention Computation (Paper Notation)
// ============================================
// This is the exact notation from Table 2 of the paper

// Step 1: Project to Q, K, V spaces
Query[p, dk] = WQ[dk, d] X[p, d]
Key[p, dk] = WK[dk, d] X[p, d]
Val[p, dv] = WV[dv, d] X[p, d]

// Step 2: Compute attention scores and normalize
// THE KEY FEATURE: p'. indicates normalization over p' for each p
// This is the normalized index feature from the paper!
//
// Without the dot (p_prime):
//   Comp[p, p_prime] would be raw scores
//
// With the dot (p_prime.):
//   Comp[p, p_prime.] applies softmax over p_prime dimension
//   Result: for each p, Σ_p' Comp[p, p'] = 1.0
//
// Paper equation (Table 2):
//   Comp[p, p'.] = softmax(Query[p, dk] Key[p', dk] / sqrt(Dk))

Comp[p, p_prime.] = softmax(Query[p, dk] Key[p_prime, dk] / sqrt_dk)

// Step 3: Compute weighted sum of values
// Comp already contains normalized attention weights
Attn[p, dv] = Comp[p, p_prime] Val[p_prime, dv]

// ============================================
// Queries: Verify Normalized Indices Behavior
// ============================================

// Query the normalized attention weights
// For each position p, these should sum to 1.0
Comp[0, 0]?  // Attention from position 0 to position 0
Comp[0, 1]?  // Attention from position 0 to position 1
Comp[0, 2]?  // Attention from position 0 to position 2

// Verify normalization: sum should equal 1.0
attention_sum_0 = Comp[0, p_prime]
attention_sum_0?  // Should be 1.0

// Check other positions
Comp[1, 0]?
Comp[1, 1]?
Comp[1, 2]?

attention_sum_1 = Comp[1, p_prime]
attention_sum_1?  // Should be 1.0

// Final attention output
Attn[0, 0]?
Attn[0, 1]?
Attn[1, 0]?
Attn[1, 1]?

// ============================================
// Key Insight from the Paper
// ============================================
// The normalized index (p_prime.) is syntactic sugar that combines:
// 1. Computing scores: Score[p, p'] = Query[p, dk] Key[p', dk] / sqrt_dk
// 2. Applying softmax: Attn[p, p'] = exp(Score[p, p']) / Σ_{p''} exp(Score[p, p''])
//
// Into a single, elegant operation:
//   Comp[p, p'.] = softmax(...)
//
// This makes attention mechanisms extremely concise and readable!
