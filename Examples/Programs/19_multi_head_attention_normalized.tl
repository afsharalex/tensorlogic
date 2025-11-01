// ============================================
// Multi-Head Attention with Normalized Indices
// ============================================
// Demonstrates the paper's notation for multi-head attention
// using normalized indices for clean, readable code
//
// Paper reference (Domingos, 2025, Table 2):
//   "Comp[b, h, p, p'.] = softmax(Query[b, h, p, dk] Key[b, h, p', dk]/sqrt(Dk))"
//
// This implements the exact notation from the paper with:
// - b: batch dimension
// - h: attention head dimension
// - p, p': position dimensions
// - p'.: normalized over p' for each (b, h, p) triple

// ============================================
// Configuration
// ============================================
// Simplified setup:
// - 1 batch (b=0)
// - 2 attention heads (h=0,1)
// - 3 sequence positions (p=0,1,2)
// - 2-dimensional queries/keys (dk=2)
// - 2-dimensional values (dv=2)

// ============================================
// Input Embeddings
// ============================================
// Input sequence: 3 positions, 4-dimensional embeddings
// We'll project these to different spaces for each head

X[0, 0] = 1.0
X[0, 1] = 0.5
X[0, 2] = 0.8
X[0, 3] = 0.3

X[1, 0] = 0.6
X[1, 1] = 1.2
X[1, 2] = 0.4
X[1, 3] = 0.9

X[2, 0] = 0.9
X[2, 1] = 0.7
X[2, 2] = 1.1
X[2, 3] = 0.2

// ============================================
// Weight Matrices (per head)
// ============================================
// Each head has its own Q, K, V projection matrices
// Shape: [heads, output_dim, input_dim]

// Head 0 - Query weights
WQ[0, 0, 0] = 0.5
WQ[0, 0, 1] = 0.3
WQ[0, 0, 2] = 0.2
WQ[0, 0, 3] = 0.1

WQ[0, 1, 0] = 0.4
WQ[0, 1, 1] = 0.6
WQ[0, 1, 2] = 0.1
WQ[0, 1, 3] = 0.2

// Head 1 - Query weights (different from Head 0)
WQ[1, 0, 0] = 0.3
WQ[1, 0, 1] = 0.5
WQ[1, 0, 2] = 0.4
WQ[1, 0, 3] = 0.3

WQ[1, 1, 0] = 0.2
WQ[1, 1, 1] = 0.4
WQ[1, 1, 2] = 0.5
WQ[1, 1, 3] = 0.6

// Head 0 - Key weights
WK[0, 0, 0] = 0.6
WK[0, 0, 1] = 0.2
WK[0, 0, 2] = 0.4
WK[0, 0, 3] = 0.3

WK[0, 1, 0] = 0.3
WK[0, 1, 1] = 0.5
WK[0, 1, 2] = 0.3
WK[0, 1, 3] = 0.4

// Head 1 - Key weights
WK[1, 0, 0] = 0.4
WK[1, 0, 1] = 0.3
WK[1, 0, 2] = 0.5
WK[1, 0, 3] = 0.2

WK[1, 1, 0] = 0.5
WK[1, 1, 1] = 0.2
WK[1, 1, 2] = 0.4
WK[1, 1, 3] = 0.5

// Head 0 - Value weights
WV[0, 0, 0] = 0.7
WV[0, 0, 1] = 0.4
WV[0, 0, 2] = 0.3
WV[0, 0, 3] = 0.2

WV[0, 1, 0] = 0.2
WV[0, 1, 1] = 0.8
WV[0, 1, 2] = 0.5
WV[0, 1, 3] = 0.6

// Head 1 - Value weights
WV[1, 0, 0] = 0.6
WV[1, 0, 1] = 0.3
WV[1, 0, 2] = 0.5
WV[1, 0, 3] = 0.4

WV[1, 1, 0] = 0.3
WV[1, 1, 1] = 0.7
WV[1, 1, 2] = 0.4
WV[1, 1, 3] = 0.8

// Scaling factor
sqrt_dk = 1.414

// ============================================
// Multi-Head Attention (Paper Notation)
// ============================================
// This follows the exact structure from Table 2 of the paper

// Project input to Q, K, V for each head
// Query[h, p, dk] = WQ[h, dk, d] X[p, d]
Query[h, p, dk] = WQ[h, dk, d] X[p, d]

// Key[h, p, dk] = WK[h, dk, d] X[p, d]
Key[h, p, dk] = WK[h, dk, d] X[p, d]

// Val[h, p, dv] = WV[h, dv, d] X[p, d]
Val[h, p, dv] = WV[h, dv, d] X[p, d]

// THE KEY FEATURE: Multi-head attention with normalized indices
// For each head h and position p, compute attention over all positions p'
// The "p'." creates a probability distribution over source positions
//
// Paper equation (adapted from Table 2):
//   Comp[h, p, p'.] = softmax(Query[h, p, dk] Key[h, p', dk] / sqrt_dk)

Comp[h, p, p_prime.] = softmax(Query[h, p, dk] Key[h, p_prime, dk] / sqrt_dk)

// Compute attention output for each head
// Weighted sum of values using normalized attention weights
Attn[h, p, dv] = Comp[h, p, p_prime] Val[h, p_prime, dv]

// ============================================
// Concatenate Heads and Project (Optional)
// ============================================
// In a full transformer, we'd concatenate attention heads
// and project through an output matrix
// For simplicity, we'll just query individual head outputs

// ============================================
// Queries: Verify Multi-Head Normalized Attention
// ============================================

// Head 0, Position 0: Attention distribution
Comp[0, 0, 0]?  // How much does head 0, pos 0 attend to pos 0?
Comp[0, 0, 1]?  // How much does head 0, pos 0 attend to pos 1?
Comp[0, 0, 2]?  // How much does head 0, pos 0 attend to pos 2?

// Verify normalization for head 0, position 0
attn_sum_h0_p0 = Comp[0, 0, p_prime]
attn_sum_h0_p0?  // Should be 1.0

// Head 1, Position 0: Different attention pattern
Comp[1, 0, 0]?
Comp[1, 0, 1]?
Comp[1, 0, 2]?

// Verify normalization for head 1, position 0
attn_sum_h1_p0 = Comp[1, 0, p_prime]
attn_sum_h1_p0?  // Should be 1.0

// Compare attention outputs from different heads
Attn[0, 0, 0]?  // Head 0 output for position 0, dimension 0
Attn[1, 0, 0]?  // Head 1 output for position 0, dimension 0

// These should be different, showing that each head
// learns different attention patterns!

// ============================================
// Key Insights
// ============================================
// 1. Normalized indices make multi-head attention EXTREMELY concise
//    - Compare to manual softmax computation in PyTorch/TensorFlow
//    - The ".\" notation is self-documenting
//
// 2. Each head can learn different attention patterns
//    - Head 0 might focus on local context
//    - Head 1 might focus on long-range dependencies
//
// 3. The normalization is automatically applied
//    - For each (head, query_position) pair
//    - Over all key positions
//    - Guaranteeing a valid probability distribution
//
// 4. This is the EXACT notation from the paper
//    - Shows how Tensor Logic makes transformers elegant
//    - Much cleaner than PyTorch's attention implementation
