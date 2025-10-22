// ============================================
// Masked Self-Attention (Decoder-style)
// ============================================
// Prevents positions from attending to future positions
// Used in language models (GPT-style)

// Input (4 positions)
X[0, 0] = 1.0
X[0, 1] = 0.5
X[1, 0] = 0.6
X[1, 1] = 1.2
X[2, 0] = 0.9
X[2, 1] = 0.7
X[3, 0] = 0.4
X[3, 1] = 0.9

// Projection matrices (simplified 2D)
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

sqrt_dk = 1.414

// Causal mask: position p can only attend to p' <= p
// Mask[p, p'] = 1 if p' <= p, else 0
Mask[0, 0] = 1.0
Mask[0, 1] = 0.0
Mask[0, 2] = 0.0
Mask[0, 3] = 0.0

Mask[1, 0] = 1.0
Mask[1, 1] = 1.0
Mask[1, 2] = 0.0
Mask[1, 3] = 0.0

Mask[2, 0] = 1.0
Mask[2, 1] = 1.0
Mask[2, 2] = 1.0
Mask[2, 3] = 0.0

Mask[3, 0] = 1.0
Mask[3, 1] = 1.0
Mask[3, 2] = 1.0
Mask[3, 3] = 1.0

// Large negative value to mask out positions
neg_inf = -1000.0

// Projections
Query[p, dk] = WQ[dk, d] X[p, d]
Key[p, dk] = WK[dk, d] X[p, d]
Value[p, dv] = WV[dv, d] X[p, d]

// Compute scores
RawScores[p, p_prime] = Query[p, dk] Key[p_prime, dk] / sqrt_dk

// Apply mask: masked positions get very negative scores
// so softmax gives them ~0 attention
MaskedScores[p, p_prime] = RawScores[p, p_prime] * Mask[p, p_prime] +
                           neg_inf * (1.0 - Mask[p, p_prime])

// Softmax (masked positions will have ~0 weight)
Attn[p, p_prime.] = softmax(MaskedScores[p, p_prime])

// Output
Output[p, dv] = Attn[p, p_prime] Value[p_prime, dv]

// Position 0 can only attend to itself
Attn[0, p_prime]?  // Should be [1.0, ~0, ~0, ~0]

// Position 2 can attend to positions 0, 1, 2
Attn[2, p_prime]?  // Should sum to 1.0, with weight on 0,1,2 only

Output[p, dv]?