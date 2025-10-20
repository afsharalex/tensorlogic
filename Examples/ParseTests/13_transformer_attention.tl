// Example 13: Transformer Self-Attention
// Simplified transformer attention mechanism

// Compute Query, Key, Value projections
Query[p, dk] = WQ[dk, d] X[p, d]
Key[p, dk] = WK[dk, d] X[p, d]
Value[p, dv] = WV[dv, d] X[p, d]

// Compute attention scores
Scores[p, p_prime] = Query[p, dk] Key[p_prime, dk] / sqrt_dk

// Apply softmax over p_prime dimension
Attn[p, p_prime.] = softmax(Scores[p, p_prime])

// Compute weighted sum of values
Output[p, dv] = Attn[p, p_prime] Value[p_prime, dv]
