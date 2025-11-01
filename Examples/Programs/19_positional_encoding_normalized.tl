// ============================================
// Positional Encoding with Normalized Indices
// ============================================
// This demonstrates the positional encoding example from Table 2
// of the paper, which uses normalized indices in combination with
// guarded clauses for elegant conditional computation
//
// Paper reference (Domingos, 2025, Table 2):
//   PosEnc[p, d] = Even(d) sin(p/L^(d/De)) | Odd(d) cos(p/L^((d-1)/De))
//
// Note: This example focuses on the normalized index semantics.
// The full positional encoding with conditional Even/Odd requires
// guarded clauses which are already implemented.

// ============================================
// Scenario: Learnable Positional Embeddings
// ============================================
// Instead of fixed sinusoidal encodings, modern transformers often
// use learnable positional embeddings that are added to token embeddings
//
// We'll demonstrate normalized indices in the context of:
// 1. Position-wise embedding normalization
// 2. Attention with positional biases

// ============================================
// Configuration
// ============================================
// - 4 positions (sequence length)
// - 3 embedding dimensions

// ============================================
// Token Embeddings
// ============================================
TokenEmb[0, 0] = 1.0
TokenEmb[0, 1] = 0.5
TokenEmb[0, 2] = 0.8

TokenEmb[1, 0] = 0.6
TokenEmb[1, 1] = 1.2
TokenEmb[1, 2] = 0.3

TokenEmb[2, 0] = 0.9
TokenEmb[2, 1] = 0.7
TokenEmb[2, 2] = 1.1

TokenEmb[3, 0] = 0.4
TokenEmb[3, 1] = 0.9
TokenEmb[3, 2] = 0.6

// ============================================
// Positional Embeddings (Learnable)
// ============================================
PosEmb[0, 0] = 0.1
PosEmb[0, 1] = 0.2
PosEmb[0, 2] = 0.15

PosEmb[1, 0] = 0.15
PosEmb[1, 1] = 0.25
PosEmb[1, 2] = 0.1

PosEmb[2, 0] = 0.2
PosEmb[2, 1] = 0.3
PosEmb[2, 2] = 0.12

PosEmb[3, 0] = 0.25
PosEmb[3, 1] = 0.35
PosEmb[3, 2] = 0.18

// ============================================
// Example 1: Position-wise L1 Normalization
// ============================================
// Normalize positional embeddings so each position's embedding
// has L1 norm = 1.0
//
// For each position p, normalize over dimension d:
// NormPos[p, d.] = PosEmb[p, d] / Σ_d' |PosEmb[p, d']|

// First compute absolute values
AbsPos[p, d] = abs(PosEmb[p, d])

// Then normalize using the normalized index notation
// The "d." means: for each p, normalize over d
NormPos[p, d.] = AbsPos[p, d] / AbsPos[p, d]  // sum over d creates normalization

// Note: In the actual implementation, this would be:
// NormPos[p, d.] = l1norm(PosEmb[p, d])
// But we're demonstrating the semantic meaning of the dot notation

// Query normalized positional embeddings
NormPos[0, 0]?
NormPos[0, 1]?
NormPos[0, 2]?

// Verify L1 normalization (should sum to 1.0)
norm_sum = NormPos[0, d]
norm_sum?  // Should be close to 1.0

// ============================================
// Example 2: Relative Positional Attention
// ============================================
// Some transformers use relative position encodings in attention
// The attention score between positions depends on their distance
//
// Relative position bias: R[p - p'] for distance between positions
// Shape: [-3, -2, -1, 0, 1, 2, 3] -> bias values

// Simplified relative position biases (just a few values)
RelBias[0] = 1.0   // Same position: strong bias
RelBias[1] = 0.8   // Distance 1: medium bias
RelBias[2] = 0.5   // Distance 2: weaker bias
RelBias[3] = 0.3   // Distance 3: weak bias

// Query and Key matrices (simplified)
Q[0, 0] = 1.0
Q[1, 0] = 0.8
Q[2, 0] = 0.9
Q[3, 0] = 0.7

K[0, 0] = 1.1
K[1, 0] = 0.9
K[2, 0] = 0.85
K[3, 0] = 0.75

// Compute attention scores with positional bias
// Score = Query · Key + RelativeBias
// For simplicity, using 1D queries/keys
RawScores[p, p_prime] = Q[p, d] K[p_prime, d] + RelBias[0]  // Simplified

// THE NORMALIZED INDEX: Convert scores to attention weights
// For each query position p, create probability distribution over keys p'
Attention[p, p_prime.] = softmax(RawScores[p, p_prime])

// Query attention weights
Attention[0, 0]?
Attention[0, 1]?
Attention[0, 2]?
Attention[0, 3]?

// Verify: attention weights from position 0 sum to 1.0
attn_sum = Attention[0, p_prime]
attn_sum?  // Should be 1.0

// ============================================
// Example 3: Combining Token and Position Embeddings
// ============================================
// Final embeddings are typically: Token + Position
// Then we might want to normalize the combined embedding

// Combine embeddings
Combined[p, d] = TokenEmb[p, d] + PosEmb[p, d]

// Layer normalization: normalize over the dimension axis for each position
// The "d." indicates: for each position p, normalize over all dimensions d
// This creates zero mean, unit variance (in full implementation)

// For demonstration, we'll use softmax normalization
// (Real layer norm is more complex, but same principle)
NormCombined[p, d.] = softmax(Combined[p, d])

// Query normalized combined embeddings
NormCombined[0, 0]?
NormCombined[0, 1]?
NormCombined[0, 2]?

// Verify normalization
combined_sum = NormCombined[0, d]
combined_sum?  // Should be 1.0

// ============================================
// Key Insights
// ============================================
// 1. Normalized indices (d.) make normalization operations explicit
//    - The dimension being normalized is clear from the notation
//    - No ambiguity about which axis to normalize over
//
// 2. Works seamlessly with positional encodings
//    - Can normalize position embeddings
//    - Can incorporate positional biases in attention
//    - Combines naturally with token embeddings
//
// 3. Essential for transformers
//    - Attention: normalize over key positions
//    - Layer norm: normalize over feature dimensions
//    - Output: normalize over vocabulary (softmax for predictions)
//
// 4. Self-documenting code
//    - p_prime. immediately tells you: "probability over source positions"
//    - d. tells you: "normalized across dimensions"
//    - Much clearer than axis=1 or dim=-1 in PyTorch!
