// ============================================
// Transformer Self-Attention Mechanism
// ============================================
// This program implements the core self-attention operation
// from "Attention Is All You Need" (Vaswani et al., 2017)
//
// Self-attention allows each position in a sequence to attend
// to all other positions, computing context-aware representations
//
// Key components:
// 1. Query, Key, Value projections
// 2. Attention score computation (scaled dot-product)
// 3. Softmax normalization
// 4. Weighted aggregation of values

// ============================================
// Input: Sequence Embeddings
// ============================================
// X represents a sequence of 4 tokens, each with 3-dimensional embeddings
// Shape: [positions, embedding_dim] = [4, 3]
//
// Example: Short sentence with 4 words
// Visual representation:
//        d=0   d=1   d=2
// p=0  [ 1.0   0.5   0.8 ]  (token 0: "The")
// p=1  [ 0.6   1.2   0.3 ]  (token 1: "cat")
// p=2  [ 0.9   0.7   1.1 ]  (token 2: "sat")
// p=3  [ 0.4   0.9   0.6 ]  (token 3: "down")

// Token 0 embedding
X[0, 0] = 1.0
X[0, 1] = 0.5
X[0, 2] = 0.8

// Token 1 embedding
X[1, 0] = 0.6
X[1, 1] = 1.2
X[1, 2] = 0.3

// Token 2 embedding
X[2, 0] = 0.9
X[2, 1] = 0.7
X[2, 2] = 1.1

// Token 3 embedding
X[3, 0] = 0.4
X[3, 1] = 0.9
X[3, 2] = 0.6

// ============================================
// Attention Parameters
// ============================================
// Dimensions:
// - d = 3 (input embedding dimension)
// - dk = 2 (query/key dimension)
// - dv = 2 (value dimension)

// Query projection matrix WQ
// Shape: [dk, d] = [2, 3]
// Projects input embeddings to query space
WQ[0, 0] = 0.5
WQ[0, 1] = 0.3
WQ[0, 2] = 0.2
WQ[1, 0] = 0.4
WQ[1, 1] = 0.6
WQ[1, 2] = 0.1

// Key projection matrix WK
// Shape: [dk, d] = [2, 3]
// Projects input embeddings to key space
WK[0, 0] = 0.6
WK[0, 1] = 0.2
WK[0, 2] = 0.4
WK[1, 0] = 0.3
WK[1, 1] = 0.5
WK[1, 2] = 0.3

// Value projection matrix WV
// Shape: [dv, d] = [2, 3]
// Projects input embeddings to value space
WV[0, 0] = 0.7
WV[0, 1] = 0.4
WV[0, 2] = 0.3
WV[1, 0] = 0.2
WV[1, 1] = 0.8
WV[1, 2] = 0.5

// Scaling factor for attention scores
// sqrt_dk = sqrt(2) ≈ 1.414
sqrt_dk = 1.414

// ============================================
// Step 1: Compute Query, Key, Value Projections
// ============================================
// Transform input embeddings into query, key, and value representations
// Each position gets its own Q, K, V vectors

// Query matrix: What this position is looking for
// Query[p, dk] = WQ[dk, d] X[p, d]
// For each position p, compute a dk-dimensional query vector
//
// Query[0, 0] = WQ[0,0]*X[0,0] + WQ[0,1]*X[0,1] + WQ[0,2]*X[0,2]
//             = 0.5*1.0 + 0.3*0.5 + 0.2*0.8
//             = 0.5 + 0.15 + 0.16 = 0.81
// Query[0, 1] = WQ[1,0]*X[0,0] + WQ[1,1]*X[0,1] + WQ[1,2]*X[0,2]
//             = 0.4*1.0 + 0.6*0.5 + 0.1*0.8
//             = 0.4 + 0.3 + 0.08 = 0.78

Query[p, dk] = WQ[dk, d] X[p, d]

// Key matrix: What this position offers
// Key[p, dk] = WK[dk, d] X[p, d]
// For each position p, compute a dk-dimensional key vector
//
// Key[0, 0] = WK[0,0]*X[0,0] + WK[0,1]*X[0,1] + WK[0,2]*X[0,2]
//           = 0.6*1.0 + 0.2*0.5 + 0.4*0.8
//           = 0.6 + 0.1 + 0.32 = 1.02
// Key[0, 1] = WK[1,0]*X[0,0] + WK[1,1]*X[0,1] + WK[1,2]*X[0,2]
//           = 0.3*1.0 + 0.5*0.5 + 0.3*0.8
//           = 0.3 + 0.25 + 0.24 = 0.79

Key[p, dk] = WK[dk, d] X[p, d]

// Value matrix: What this position contains
// Value[p, dv] = WV[dv, d] X[p, d]
// For each position p, compute a dv-dimensional value vector
//
// Value[0, 0] = WV[0,0]*X[0,0] + WV[0,1]*X[0,1] + WV[0,2]*X[0,2]
//             = 0.7*1.0 + 0.4*0.5 + 0.3*0.8
//             = 0.7 + 0.2 + 0.24 = 1.14
// Value[0, 1] = WV[1,0]*X[0,0] + WV[1,1]*X[0,1] + WV[1,2]*X[0,2]
//             = 0.2*1.0 + 0.8*0.5 + 0.5*0.8
//             = 0.2 + 0.4 + 0.4 = 1.0

Value[p, dv] = WV[dv, d] X[p, d]

// ============================================
// Step 2: Compute Attention Scores
// ============================================
// Calculate how much each position should attend to every other position
// Using scaled dot-product attention
//
// Scores[p, p_prime] = (Query[p] · Key[p_prime]) / sqrt(dk)
//
// The score measures compatibility between position p's query
// and position p_prime's key
//
// Example calculation for Scores[0, 0]:
// Scores[0, 0] = (Query[0, 0]*Key[0, 0] + Query[0, 1]*Key[0, 1]) / sqrt_dk
//              = (Query[0] · Key[0]) / 1.414
//              = (0.81*1.02 + 0.78*0.79) / 1.414
//              = (0.8262 + 0.6162) / 1.414
//              = 1.4424 / 1.414
//              ≈ 1.020
//
// Scores[0, 1] = (Query[0] · Key[1]) / sqrt_dk
// ... and so on for all position pairs

Scores[p, p_prime] = Query[p, dk] Key[p_prime, dk] / sqrt_dk

// ============================================
// Step 3: Apply Softmax Normalization
// ============================================
// Convert scores to attention weights (probability distribution)
// The dot notation "p_prime." indicates that softmax normalizes
// over the p_prime dimension for each p
//
// For each query position p, the attention weights over all
// key positions p_prime sum to 1.0
//
// Attn[p, p_prime] = exp(Scores[p, p_prime]) / Σ_{p'} exp(Scores[p, p'])
//
// Example for position 0:
// Attn[0, 0] = exp(Scores[0,0]) / (exp(Scores[0,0]) + exp(Scores[0,1]) +
//                                   exp(Scores[0,2]) + exp(Scores[0,3]))
//
// This creates a probability distribution: Attn[0, :] sums to 1.0
// Higher attention weights mean position 0 attends more to that position

Attn[p, p_prime.] = softmax(Scores[p, p_prime])

// ============================================
// Step 4: Compute Weighted Sum of Values
// ============================================
// Aggregate value vectors using attention weights
// This produces the final context-aware representation for each position
//
// Output[p, dv] = Σ_{p'} Attn[p, p'] * Value[p', dv]
//
// For each position p and each value dimension dv:
// Output[p, dv] = Attn[p, 0]*Value[0, dv] +
//                 Attn[p, 1]*Value[1, dv] +
//                 Attn[p, 2]*Value[2, dv] +
//                 Attn[p, 3]*Value[3, dv]
//
// This is a weighted average of all value vectors,
// where weights come from attention scores
//
// Example for Output[0, 0]:
// Output[0, 0] = Attn[0,0]*Value[0,0] + Attn[0,1]*Value[1,0] +
//                Attn[0,2]*Value[2,0] + Attn[0,3]*Value[3,0]
//
// The output incorporates information from all positions,
// weighted by how much attention position 0 gives to each

Output[p, dv] = Attn[p, p_prime] Value[p_prime, dv]

// ============================================
// Queries: Inspect Intermediate Results
// ============================================

// Query projections for position 0
Query[0, 0]?  // Query vector for token 0, dimension 0
Query[0, 1]?  // Query vector for token 0, dimension 1

// Key projections for position 0
Key[0, 0]?
Key[0, 1]?

// Value projections for position 0
Value[0, 0]?
Value[0, 1]?

// Attention scores (before softmax)
Scores[0, 0]?  // How much does position 0 attend to itself?
Scores[0, 1]?  // How much does position 0 attend to position 1?
Scores[0, 2]?  // How much does position 0 attend to position 2?
Scores[0, 3]?  // How much does position 0 attend to position 3?

// Attention weights (after softmax) - should sum to 1.0
Attn[0, 0]?
Attn[0, 1]?
Attn[0, 2]?
Attn[0, 3]?

// Verify attention weights sum to 1.0
attention_sum = Attn[0, p_prime]
attention_sum?  // Should return: 1.0

// Final output for position 0
Output[0, 0]?
Output[0, 1]?

// Query all outputs
Output[p, dv]?