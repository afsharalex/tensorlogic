// ============================================
// Cross-Attention with Normalized Indices
// ============================================
// Demonstrates normalized indices in encoder-decoder attention,
// where queries come from the decoder and keys/values from the encoder
//
// This is used in:
// - Machine translation
// - Image captioning
// - Sequence-to-sequence models
//
// Key concept: Decoder attends to encoder outputs using normalized
// attention weights over encoder positions

// ============================================
// Configuration
// ============================================
// Encoder: 4 positions (source sequence)
// Decoder: 3 positions (target sequence)
// Dimensions: 2D for simplicity

// ============================================
// Encoder Outputs (Keys and Values)
// ============================================
// These come from the encoder and represent the source sequence
// Example: French sentence "Le chat noir dort"
//   Position 0: "Le"
//   Position 1: "chat"
//   Position 2: "noir"
//   Position 3: "dort"

EncOut[0, 0] = 1.0
EncOut[0, 1] = 0.5

EncOut[1, 0] = 0.8
EncOut[1, 1] = 0.9

EncOut[2, 0] = 0.6
EncOut[2, 1] = 1.1

EncOut[3, 0] = 0.9
EncOut[3, 1] = 0.7

// ============================================
// Decoder States (Queries)
// ============================================
// These come from the decoder and represent the target sequence so far
// Example: English translation "The black cat"
//   Position 0: "The"
//   Position 1: "black"
//   Position 2: "cat"

DecState[0, 0] = 0.9
DecState[0, 1] = 0.6

DecState[1, 0] = 0.7
DecState[1, 1] = 1.0

DecState[2, 0] = 0.8
DecState[2, 1] = 0.8

// ============================================
// Cross-Attention Parameters
// ============================================
// Weight matrices for projecting decoder and encoder representations

// Query projection (from decoder)
WQ[0, 0] = 0.5
WQ[0, 1] = 0.3
WQ[1, 0] = 0.4
WQ[1, 1] = 0.6

// Key projection (from encoder)
WK[0, 0] = 0.6
WK[0, 1] = 0.2
WK[1, 0] = 0.3
WK[1, 1] = 0.5

// Value projection (from encoder)
WV[0, 0] = 0.7
WV[0, 1] = 0.4
WV[1, 0] = 0.2
WV[1, 1] = 0.8

sqrt_dk = 1.414

// ============================================
// Cross-Attention Computation
// ============================================
// Key difference from self-attention:
// - Queries come from DECODER (target)
// - Keys and Values come from ENCODER (source)

// Project decoder states to query space
// q: decoder/target position
Query[q, dk] = WQ[dk, d] DecState[q, d]

// Project encoder outputs to key space
// s: encoder/source position
Key[s, dk] = WK[dk, d] EncOut[s, d]

// Project encoder outputs to value space
Value[s, dv] = WV[dv, d] EncOut[s, d]

// Compute cross-attention scores
// For each decoder position q, score all encoder positions s
RawScores[q, s] = Query[q, dk] Key[s, dk] / sqrt_dk

// THE NORMALIZED INDEX: Create attention distribution
// For each decoder position q, normalize over encoder positions s
// This tells us: "how much should decoder position q attend to
//                 each encoder position s?"
//
// The "s." indicates: normalize over source positions
CrossAttn[q, s.] = softmax(RawScores[q, s])

// Compute context vector for each decoder position
// Weighted sum of encoder values, using cross-attention weights
Context[q, dv] = CrossAttn[q, s] Value[s, dv]

// ============================================
// Queries: Inspect Cross-Attention
// ============================================

// Decoder position 0 ("The") attending to encoder
CrossAttn[0, 0]?  // Attention to "Le"
CrossAttn[0, 1]?  // Attention to "chat"
CrossAttn[0, 2]?  // Attention to "noir"
CrossAttn[0, 3]?  // Attention to "dort"

// Should sum to 1.0 (probability distribution over source)
attn_sum_q0 = CrossAttn[0, s]
attn_sum_q0?  // Should be 1.0

// Decoder position 1 ("black") attending to encoder
// Likely should attend strongly to "noir" (position 2)
CrossAttn[1, 0]?
CrossAttn[1, 1]?
CrossAttn[1, 2]?  // Expected to be high for "noir"
CrossAttn[1, 3]?

attn_sum_q1 = CrossAttn[1, s]
attn_sum_q1?  // Should be 1.0

// Decoder position 2 ("cat") attending to encoder
// Likely should attend strongly to "chat" (position 1)
CrossAttn[2, 0]?
CrossAttn[2, 1]?  // Expected to be high for "chat"
CrossAttn[2, 2]?
CrossAttn[2, 3]?

attn_sum_q2 = CrossAttn[2, s]
attn_sum_q2?  // Should be 1.0

// Context vectors (decoder representations enhanced with encoder information)
Context[0, 0]?
Context[0, 1]?
Context[1, 0]?
Context[1, 1]?
Context[2, 0]?
Context[2, 1]?

// ============================================
// Comparison: Self-Attention vs Cross-Attention
// ============================================

// For comparison, compute self-attention within decoder
// (This would be in a different layer)

// Self-attention: decoder attends to itself
SelfQuery[q, dk] = WQ[dk, d] DecState[q, d]
SelfKey[q_prime, dk] = WK[dk, d] DecState[q_prime, d]

SelfScores[q, q_prime] = SelfQuery[q, dk] SelfKey[q_prime, dk] / sqrt_dk

// THE KEY DIFFERENCE:
// Self-attention: q_prime. (normalize over decoder positions)
// Cross-attention: s. (normalize over encoder positions)

SelfAttn[q, q_prime.] = softmax(SelfScores[q, q_prime])

// Query self-attention weights
SelfAttn[0, 0]?  // How much does decoder pos 0 attend to itself?
SelfAttn[0, 1]?  // How much does decoder pos 0 attend to pos 1?
SelfAttn[0, 2]?  // How much does decoder pos 0 attend to pos 2?

self_attn_sum = SelfAttn[0, q_prime]
self_attn_sum?  // Should be 1.0

// ============================================
// Key Insights
// ============================================
// 1. Cross-attention: Normalized indices over SOURCE positions
//    - CrossAttn[q, s.] means: for each target position q,
//      create probability distribution over source positions s
//
// 2. Self-attention: Normalized indices over SAME sequence
//    - SelfAttn[q, q'.] means: for each target position q,
//      create probability distribution over target positions q'
//
// 3. The dot notation makes the semantic difference crystal clear:
//    - s. = "attending over source sequence"
//    - q'. = "attending within query sequence"
//
// 4. Essential for encoder-decoder models:
//    - Translation: target word attends to source words
//    - Summarization: summary sentence attends to document
//    - Image captioning: caption word attends to image regions
//
// 5. Normalized indices eliminate ambiguity:
//    - No confusion about which dimension is being normalized
//    - Self-documenting code for complex attention patterns
