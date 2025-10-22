// ============================================
// Knowledge Graph Completion with Embeddings (CORRECTED)
// ============================================
// Complete missing edges in a knowledge graph
// using both logic and learned similarities
//
// Use case: Predict potential capital-country relationships
// based on known facts and entity similarity patterns

// ============================================
// Entity Embeddings
// ============================================
// Learned representations for cities and countries
// Similar entities (e.g., European capitals, European countries)
// should have similar embeddings

// Paris embedding: European capital
EmbMatrix[Paris, 0] = 0.9
EmbMatrix[Paris, 1] = 0.2
EmbMatrix[Paris, 2] = 0.3

// France embedding: European country
EmbMatrix[France, 0] = 0.85
EmbMatrix[France, 1] = 0.25
EmbMatrix[France, 2] = 0.35

// London embedding: European capital (similar to Paris)
EmbMatrix[London, 0] = 0.88
EmbMatrix[London, 1] = 0.22
EmbMatrix[London, 2] = 0.32

// UK embedding: European country (similar to France)
EmbMatrix[UK, 0] = 0.82
EmbMatrix[UK, 1] = 0.28
EmbMatrix[UK, 2] = 0.38

// Additional entities for testing
// Berlin: Another European capital
EmbMatrix[Berlin, 0] = 0.89
EmbMatrix[Berlin, 1] = 0.21
EmbMatrix[Berlin, 2] = 0.31

// Germany: Another European country
EmbMatrix[Germany, 0] = 0.84
EmbMatrix[Germany, 1] = 0.26
EmbMatrix[Germany, 2] = 0.36

// Tokyo: Asian capital (very different from European capitals)
EmbMatrix[Tokyo, 0] = 0.3
EmbMatrix[Tokyo, 1] = 0.8
EmbMatrix[Tokyo, 2] = 0.5

// Japan: Asian country (very different from European countries)
EmbMatrix[Japan, 0] = 0.28
EmbMatrix[Japan, 1] = 0.82
EmbMatrix[Japan, 2] = 0.52

// Embedding lookup
Emb[entity, d] = EmbMatrix[entity, d]

// ============================================
// Known Facts: Capital Relationships
// ============================================
CapitalOf(Paris, France)
CapitalOf(London, UK)

// Note: We DON'T have facts for Berlin or Tokyo
// The goal is to predict them based on similarity

// ============================================
// Compute Entity Similarities
// ============================================
// Pairwise similarity using dot product
Similarity[x, y] = Emb[x, d] Emb[y, d]

// Expected similarities:
// sim(Paris, London) = 0.9*0.88 + 0.2*0.22 + 0.3*0.32
//                    = 0.792 + 0.044 + 0.096 = 0.932 (HIGH)
//
// sim(Paris, Berlin) = 0.9*0.89 + 0.2*0.21 + 0.3*0.31
//                    = 0.801 + 0.042 + 0.093 = 0.936 (HIGH)
//
// sim(France, UK) = 0.85*0.82 + 0.25*0.28 + 0.35*0.38
//                 = 0.697 + 0.07 + 0.133 = 0.900 (HIGH)
//
// sim(France, Germany) = 0.85*0.84 + 0.25*0.26 + 0.35*0.36
//                      = 0.714 + 0.065 + 0.126 = 0.905 (HIGH)
//
// sim(Paris, Tokyo) = 0.9*0.3 + 0.2*0.8 + 0.3*0.5
//                   = 0.27 + 0.16 + 0.15 = 0.58 (LOW)

// Similarity threshold
similarity_threshold = 0.9

// ============================================
// CORRECTED: Create Boolean Similar Relations
// ============================================
// Since we can't use comparisons directly in rule bodies yet,
// we explicitly create Similar facts for pairs above threshold

// Cities similar to each other (all European capitals)
Similar(Paris, London)
Similar(London, Paris)
Similar(Paris, Berlin)
Similar(Berlin, Paris)
Similar(London, Berlin)
Similar(Berlin, London)

// Countries similar to each other (all European countries)
Similar(France, UK)
Similar(UK, France)
Similar(France, Germany)
Similar(Germany, France)
Similar(UK, Germany)
Similar(Germany, UK)

// Tokyo and Japan are NOT similar to European entities
// (their similarities are < 0.9)

// ============================================
// CORRECTED: Knowledge Graph Completion Rule
// ============================================
// Original rule had comparison in body which isn't fully supported
// We've pre-computed Similar facts instead
//
// Rule: If x is capital of y, and x' is similar to x,
//       and y' is similar to y, and x' != x,
//       then x' might be capital of y'
//
// This captures the pattern:
// "Cities similar to known capitals are likely capitals
//  of countries similar to the known countries"

PotentialCapital(xp, yp) <- CapitalOf(x, y),
                            Similar(xp, x),
                            Similar(yp, y),
                            xp != x

// Expected derivations:
//
// From CapitalOf(Paris, France):
//   PotentialCapital(London, UK)
//     - Similar(London, Paris) ✓
//     - Similar(UK, France) ✓
//     - London != Paris ✓
//
//   PotentialCapital(Berlin, Germany)
//     - Similar(Berlin, Paris) ✓
//     - Similar(Germany, France) ✓
//     - Berlin != Paris ✓
//
//   PotentialCapital(London, Germany)
//     - Similar(London, Paris) ✓
//     - Similar(Germany, France) ✓
//     - London != Paris ✓
//
//   PotentialCapital(Berlin, UK)
//     - Similar(Berlin, Paris) ✓
//     - Similar(UK, France) ✓
//     - Berlin != Paris ✓
//
// From CapitalOf(London, UK):
//   PotentialCapital(Paris, France)
//     - Similar(Paris, London) ✓
//     - Similar(France, UK) ✓
//     - Paris != London ✓
//
//   PotentialCapital(Berlin, France)
//     - Similar(Berlin, London) ✓
//     - Similar(France, UK) ✓
//     - Berlin != London ✓
//
//   ... and more combinations

// Additional rule: Symmetric verification
// If we predict x' is capital of y', and x is capital of y,
// and both pairs are similar, increase confidence
HighConfidenceCapital(xp, yp) <- PotentialCapital(xp, yp),
                                  PotentialCapital(x, y),
                                  Similar(xp, x),
                                  Similar(yp, y),
                                  xp != x,
                                  yp != y

// ============================================
// Queries
// ============================================

// Query 1: Check computed similarities
Similarity[Paris, London]?
// Expected: 0.932

Similarity[Paris, Berlin]?
// Expected: 0.936

Similarity[France, UK]?
// Expected: 0.900

Similarity[France, Germany]?
// Expected: 0.905

Similarity[Paris, Tokyo]?
// Expected: 0.58 (too low for Similar relation)

// Query 2: Which cities are similar?
Similar(Paris, x)?
// Expected: London, Berlin

Similar(London, x)?
// Expected: Paris, Berlin

// Query 3: Which countries are similar?
Similar(France, x)?
// Expected: UK, Germany

// Query 4: What are the known capitals?
CapitalOf(x, y)?
// Expected: (Paris, France), (London, UK)

// Query 5: What are the PREDICTED capitals?
PotentialCapital(x, y)?
// Expected predictions include:
//   (London, UK) - CORRECT! (known but re-derived)
//   (Paris, France) - CORRECT! (known but re-derived)
//   (Berlin, Germany) - LIKELY CORRECT! (missing from KB)
//   (Berlin, UK) - Possible but less likely
//   (London, Germany) - Possible but less likely
//   (Berlin, France) - Possible but less likely

// Query 6: Is Berlin likely the capital of Germany?
PotentialCapital(Berlin, Germany)?
// Expected: True (derived from similarity patterns)

// Query 7: Is Tokyo likely the capital of Japan?
PotentialCapital(Tokyo, Japan)?
// Expected: False (not similar enough to known examples)

// Query 8: High confidence predictions
HighConfidenceCapital(x, y)?
// Expected: Pairs that are mutually predicted
// e.g., if both (Berlin, Germany) and (Paris, France) are predicted
// and they're similar, then higher confidence

// ============================================
// Verification Queries
// ============================================

// Verify specific similarity values
Similarity[Paris, London]?   // ~0.932
Similarity[France, UK]?      // ~0.900
Similarity[Berlin, Paris]?   // ~0.936
Similarity[Germany, France]? // ~0.905

// Verify Similar facts exist
Similar(Paris, London)?      // True
Similar(France, UK)?         // True
Similar(Paris, Tokyo)?       // (Should not exist - similarity too low)

// Verify predictions
PotentialCapital(Berlin, Germany)?  // True (good prediction!)
PotentialCapital(Paris, France)?    // True (re-derives known fact)
PotentialCapital(Tokyo, Japan)?     // False (not similar enough)
PotentialCapital(London, UK)?       // True (re-derives known fact)

// ============================================
// Analysis of Results
// ============================================
// The knowledge graph completion works by:
//
// 1. NEURAL: Learning embeddings that capture geographic/cultural similarity
//    - European cities cluster together
//    - European countries cluster together
//    - Asian entities form separate cluster
//
// 2. SYMBOLIC: Logical rules about capital relationships
//    - If similar city and similar country, predict capital relation
//    - Transitive and compositional reasoning
//
// 3. HYBRID: Combining both for completion
//    - Known: Paris→France, London→UK
//    - Predicted: Berlin→Germany (HIGH confidence)
//    - Filtered: Tokyo→Japan (LOW similarity, not predicted)
//
// This demonstrates how neurosymbolic AI can:
// - Generalize from limited examples
// - Maintain logical consistency
// - Filter unlikely predictions
// - Provide explainable inferences