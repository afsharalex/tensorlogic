// ============================================
// Neurosymbolic Relation Prediction (CORRECTED)
// ============================================
// Use embeddings + rules to predict unknown relationships
// with confidence scores

// ============================================
// Entity Embeddings
// ============================================
// Learned representations for 4 entities
// (Removed Dave since not in original example)

EmbMatrix[Alice, 0] = 0.8
EmbMatrix[Alice, 1] = 0.3
EmbMatrix[Alice, 2] = 0.5

EmbMatrix[Bob, 0] = 0.7
EmbMatrix[Bob, 1] = 0.4
EmbMatrix[Bob, 2] = 0.6

EmbMatrix[Charlie, 0] = 0.6
EmbMatrix[Charlie, 1] = 0.5
EmbMatrix[Charlie, 2] = 0.7

EmbMatrix[Eve, 0] = 0.75
EmbMatrix[Eve, 1] = 0.35
EmbMatrix[Eve, 2] = 0.55

// Embedding lookup
Emb[entity, d] = EmbMatrix[entity, d]

// ============================================
// Symbolic Component: Known Facts
// ============================================
Parent(Alice, Bob)
Parent(Bob, Charlie)

// ============================================
// Neural Component: Similarity Computation
// ============================================
// Compute pairwise similarity scores using dot product
SimilarityScore[x, y] = Emb[x, d] Emb[y, d]

// Expected similarity scores:
// sim(Alice, Alice) = 0.8² + 0.3² + 0.5² = 0.64 + 0.09 + 0.25 = 0.98
// sim(Alice, Bob) = 0.8*0.7 + 0.3*0.4 + 0.5*0.6 = 0.56 + 0.12 + 0.30 = 0.98
// sim(Alice, Charlie) = 0.8*0.6 + 0.3*0.5 + 0.5*0.7 = 0.48 + 0.15 + 0.35 = 0.98
// sim(Alice, Eve) = 0.8*0.75 + 0.3*0.35 + 0.5*0.55 = 0.60 + 0.105 + 0.275 = 0.98
// sim(Eve, Alice) = 0.98 (same, symmetric)
// sim(Eve, Bob) = 0.75*0.7 + 0.35*0.4 + 0.55*0.6 = 0.525 + 0.14 + 0.33 = 0.995
// sim(Eve, Charlie) = 0.75*0.6 + 0.35*0.5 + 0.55*0.7 = 0.45 + 0.175 + 0.385 = 1.01

// ============================================
// Hybrid Component: Soft Similarity Confidence
// ============================================
// Convert similarity scores to confidence values using sigmoid
// Sigmoid centers around 0.5, so scores > 0.5 get high confidence
//
// For our scores (most are ~0.98-1.0):
// sigmoid(0.98 - 0.5) = sigmoid(0.48) ≈ 0.618
// sigmoid(1.0 - 0.5) = sigmoid(0.5) ≈ 0.622

SimilarityConfidence[x, y] = sigmoid(SimilarityScore[x, y] - 0.5)

// ============================================
// CORRECTED: Confidence in Potential Parent Relationship
// ============================================
// Problem in original: Parent(y, z) is a boolean relation,
// can't be used directly in tensor multiplication
//
// Solution: We need to explicitly enumerate the predictions
// for specific entity pairs

// Convert Parent facts to numeric tensor for multiplication
// ParentTensor[x, y] = 1 if Parent(x, y), else 0
ParentTensor[Alice, Bob] = 1.0
ParentTensor[Bob, Charlie] = 1.0
ParentTensor[Alice, Charlie] = 0.0
ParentTensor[Alice, Alice] = 0.0
ParentTensor[Bob, Bob] = 0.0
ParentTensor[Bob, Alice] = 0.0
ParentTensor[Charlie, Alice] = 0.0
ParentTensor[Charlie, Bob] = 0.0
ParentTensor[Charlie, Charlie] = 0.0
ParentTensor[Eve, Alice] = 0.0
ParentTensor[Eve, Bob] = 0.0
ParentTensor[Eve, Charlie] = 0.0
ParentTensor[Alice, Eve] = 0.0
ParentTensor[Bob, Eve] = 0.0
ParentTensor[Charlie, Eve] = 0.0
ParentTensor[Eve, Eve] = 0.0

// Compute confidence that x is parent of z
// by aggregating evidence from similar entities
//
// ParentConfidence[x, z] = Σ_y SimilarityConfidence[x, y] * ParentTensor[y, z]
//
// Interpretation: x is likely parent of z if x is similar to
// someone (y) who is actually parent of z

ParentConfidence[x, z] = SimilarityConfidence[x, y] ParentTensor[y, z]

// Expected calculation for ParentConfidence[Eve, Alice]:
// = SimilarityConfidence[Eve, Alice] * ParentTensor[Alice, Alice] +
//   SimilarityConfidence[Eve, Bob] * ParentTensor[Bob, Alice] +
//   SimilarityConfidence[Eve, Charlie] * ParentTensor[Charlie, Alice] +
//   SimilarityConfidence[Eve, Eve] * ParentTensor[Eve, Alice]
// = conf(Eve,Alice)*0 + conf(Eve,Bob)*0 + conf(Eve,Charlie)*0 + conf(Eve,Eve)*0
// = 0
// (No one is parent of Alice in our KB, so confidence is 0)
//
// But wait - we want to know if Eve could be Alice's parent!
// So we should check ParentConfidence[Eve, Alice] which measures
// "how likely is Eve to be parent of Alice based on similarity to known parents"

// ============================================
// Ancestor Rules (Standard)
// ============================================
Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// ============================================
// Queries
// ============================================

// Query 1: Similarity scores (these definitely work)
SimilarityScore[Alice, Eve]?
// Expected: 0.98

SimilarityScore[Eve, Bob]?
// Expected: 0.995

SimilarityScore[Bob, Charlie]?
// Expected: 1.04

// Query 2: Similarity confidence values
SimilarityConfidence[Alice, Eve]?
// Expected: sigmoid(0.98 - 0.5) = sigmoid(0.48) ≈ 0.618

SimilarityConfidence[Eve, Alice]?
// Expected: same as above ≈ 0.618

SimilarityConfidence[Eve, Bob]?
// Expected: sigmoid(0.995 - 0.5) = sigmoid(0.495) ≈ 0.621

// Query 3: Parent confidence predictions
// Predict: Is Eve likely to be parent of Bob?
ParentConfidence[Eve, Bob]?
// Calculation:
// = SimilarityConfidence[Eve, Alice] * ParentTensor[Alice, Bob] +
//   SimilarityConfidence[Eve, Bob] * ParentTensor[Bob, Bob] +
//   SimilarityConfidence[Eve, Charlie] * ParentTensor[Charlie, Bob] +
//   SimilarityConfidence[Eve, Eve] * ParentTensor[Eve, Bob]
// = 0.618 * 1.0 + 0.621 * 0 + ~0.622 * 0 + ~0.620 * 0
// = 0.618
// HIGH confidence! Eve is similar to Alice who IS parent of Bob

// Query 4: Is Eve likely to be parent of Charlie?
ParentConfidence[Eve, Charlie]?
// Calculation:
// = SimilarityConfidence[Eve, Alice] * ParentTensor[Alice, Charlie] +
//   SimilarityConfidence[Eve, Bob] * ParentTensor[Bob, Charlie] +
//   ...
// = 0.618 * 0 + 0.621 * 1.0 + ...
// = 0.621
// HIGH confidence! Eve is similar to Bob who IS parent of Charlie

// Query 5: Is Alice likely to be parent of Charlie?
ParentConfidence[Alice, Charlie]?
// Calculation:
// = SimilarityConfidence[Alice, Alice] * ParentTensor[Alice, Charlie] +
//   SimilarityConfidence[Alice, Bob] * ParentTensor[Bob, Charlie] +
//   ...
// = ~0.618 * 0 + 0.618 * 1.0 + ...
// = 0.618
// HIGH confidence! Alice is similar to Bob who IS parent of Charlie
// (This is actually TRUE - Alice is grandparent via the transitive relation!)

// Query 6: Check ancestor relations (symbolic reasoning)
Ancestor(Alice, Charlie)?
// Expected: True (derived via transitivity)

Ancestor(Eve, Bob)?
// Expected: False (not in symbolic KB)
// But ParentConfidence[Eve, Bob] suggests it could be true!

// ============================================
// Expected Insights
// ============================================
// The neurosymbolic approach discovers:
// 1. Eve likely related to Bob (confidence 0.618)
// 2. Eve likely related to Charlie (confidence 0.621)
// 3. Alice connected to Charlie (confidence 0.618, which is actually true!)
//
// This combines:
// - Neural: Learned similarity patterns
// - Symbolic: Logical parent/ancestor structure
// - Hybrid: Probabilistic relation predictions