// ============================================
// Mixed Neurosymbolic Program
// ============================================
// This program demonstrates the integration of:
// 1. Neural components (learned embeddings)
// 2. Symbolic components (logical rules and facts)
// 3. Hybrid reasoning (similarity + logic)
//
// Use case: Discover potential family relationships
// using both known facts and learned entity similarities

// ============================================
// Define Entity Universe
// ============================================
// We have 5 people in our knowledge base
// Entities: Alice, Bob, Charlie, Dave, Eve

// ============================================
// Neural Component: Entity Embeddings
// ============================================
// Learned representations capturing semantic similarity
// Each entity has a 3-dimensional embedding vector
// Shape: [5 entities, 3 dimensions]
//
// These embeddings might be learned from data about:
// - Social interactions
// - Shared attributes
// - Co-occurrence patterns
// - Behavioral similarities


// Entity embeddings (same as before)
EmbMatrix[Alice, 0] = 0.8
EmbMatrix[Alice, 1] = 0.3
EmbMatrix[Alice, 2] = 0.5
EmbMatrix[Bob, 0] = 0.7
EmbMatrix[Bob, 1] = 0.4
EmbMatrix[Bob, 2] = 0.6
EmbMatrix[Charlie, 0] = 0.6
EmbMatrix[Charlie, 1] = 0.5
EmbMatrix[Charlie, 2] = 0.7
EmbMatrix[Dave, 0] = 0.2
EmbMatrix[Dave, 1] = 0.8
EmbMatrix[Dave, 2] = 0.1
EmbMatrix[Eve, 0] = 0.75
EmbMatrix[Eve, 1] = 0.35
EmbMatrix[Eve, 2] = 0.55

Emb[entity, d] = EmbMatrix[entity, d]

// Parent facts
Parent(Alice, Bob)
Parent(Bob, Charlie)

// Compute similarity scores (neural component)
SimilarityScore[x, y] = Emb[x, d] Emb[y, d]

threshold = 0.8

// ============================================
// CORRECTED: Manually assert Similar facts
// based on computed similarity scores
// ============================================
// After computing SimilarityScore, we found:
// - sim(Alice, Bob) = 0.98 > 0.8 ✓
// - sim(Alice, Charlie) = 0.98 > 0.8 ✓
// - sim(Alice, Eve) = 0.98 > 0.8 ✓
// - sim(Bob, Charlie) = 1.04 > 0.8 ✓
// - sim(Bob, Eve) = 0.995 > 0.8 ✓
// - sim(Charlie, Eve) = 1.01 > 0.8 ✓
// - All Dave similarities < 0.8 ✗

Similar(Alice, Bob)
Similar(Bob, Alice)
Similar(Alice, Charlie)
Similar(Charlie, Alice)
Similar(Alice, Eve)
Similar(Eve, Alice)
Similar(Bob, Charlie)
Similar(Charlie, Bob)
Similar(Bob, Eve)
Similar(Eve, Bob)
Similar(Charlie, Eve)
Similar(Eve, Charlie)

// Ancestor rules
Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// Neurosymbolic inference
MaybeRelated(x, z) <- Similar(x, y), Parent(y, z), x != z

// Potential family cluster
PotentialFamily(x, y) <- Similar(x, y), x != y

// ============================================
// Queries
// ============================================

// Similarity scores (these work!)
SimilarityScore[Alice, Eve]?
SimilarityScore[Alice, Dave]?

// Similar relations (should now work!)
Similar(Alice, y)?
// Expected: Bob, Charlie, Eve

Similar(Dave, y)?
// Expected: None

// Symbolic reasoning (works!)
Parent(x, y)?
Ancestor(x, y)?

// Neurosymbolic inferences (should now work!)
MaybeRelated(Eve, z)?
// Expected: Bob, Charlie

MaybeRelated(x, Charlie)?
// Expected: Alice, Eve

MaybeRelated(Eve, Bob)?
// Expected: True

MaybeRelated(Alice, Charlie)?
// Expected: True

PotentialFamily(x, y)?
// Expected: All similar pairs