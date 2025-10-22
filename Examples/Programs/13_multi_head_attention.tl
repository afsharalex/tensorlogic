// ============================================
// Multi-Head Self-Attention
// ============================================
// Extends single-head attention to multiple parallel attention heads
// Each head can focus on different aspects of the input

// Input (same as before)
X[0, 0] = 1.0
X[0, 1] = 0.5
X[0, 2] = 0.8
X[1, 0] = 0.6
X[1, 1] = 1.2
X[1, 2] = 0.3
X[2, 0] = 0.9
X[2, 1] = 0.7
X[2, 2] = 1.1
X[3, 0] = 0.4
X[3, 1] = 0.9
X[3, 2] = 0.6

// Parameters
sqrt_dk = 1.414

// Head 1 parameters
WQ1[0, 0] = 0.5
WQ1[0, 1] = 0.3
WQ1[0, 2] = 0.2
WQ1[1, 0] = 0.4
WQ1[1, 1] = 0.6
WQ1[1, 2] = 0.1

WK1[0, 0] = 0.6
WK1[0, 1] = 0.2
WK1[0, 2] = 0.4
WK1[1, 0] = 0.3
WK1[1, 1] = 0.5
WK1[1, 2] = 0.3

WV1[0, 0] = 0.7
WV1[0, 1] = 0.4
WV1[0, 2] = 0.3
WV1[1, 0] = 0.2
WV1[1, 1] = 0.8
WV1[1, 2] = 0.5

// Head 2 parameters (different from head 1)
WQ2[0, 0] = 0.3
WQ2[0, 1] = 0.5
WQ2[0, 2] = 0.4
WQ2[1, 0] = 0.6
WQ2[1, 1] = 0.2
WQ2[1, 2] = 0.3

WK2[0, 0] = 0.4
WK2[0, 1] = 0.6
WK2[0, 2] = 0.2
WK2[1, 0] = 0.5
WK2[1, 1] = 0.3
WK2[1, 2] = 0.4

WV2[0, 0] = 0.5
WV2[0, 1] = 0.6
WV2[0, 2] = 0.4
WV2[1, 0] = 0.4
WV2[1, 1] = 0.3
WV2[1, 2] = 0.7

// Head 1 attention
Query1[p, dk] = WQ1[dk, d] X[p, d]
Key1[p, dk] = WK1[dk, d] X[p, d]
Value1[p, dv] = WV1[dv, d] X[p, d]
Scores1[p, p_prime] = Query1[p, dk] Key1[p_prime, dk] / sqrt_dk
Attn1[p, p_prime.] = softmax(Scores1[p, p_prime])
Output1[p, dv] = Attn1[p, p_prime] Value1[p_prime, dv]

// Head 2 attention
Query2[p, dk] = WQ2[dk, d] X[p, d]
Key2[p, dk] = WK2[dk, d] X[p, d]
Value2[p, dv] = WV2[dv, d] X[p, d]
Scores2[p, p_prime] = Query2[p, dk] Key2[p_prime, dk] / sqrt_dk
Attn2[p, p_prime.] = softmax(Scores2[p, p_prime])
Output2[p, dv] = Attn2[p, p_prime] Value2[p_prime, dv]

// Concatenate heads (simplified - just add for demonstration)
// In practice, would concatenate along feature dimension
MultiHeadOutput[p, dv] = Output1[p, dv] + Output2[p, dv]

// Query results from different heads
Attn1[0, p_prime]?  // Head 1 attention pattern for position 0
Attn2[0, p_prime]?  // Head 2 attention pattern for position 0

Output1[0, dv]?
Output2[0, dv]?
MultiHeadOutput[0, dv]?