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

// Alice's embedding: [0.8, 0.3, 0.5]
// Represents Alice's "semantic position" in embedding space
EmbMatrix[Alice, 0] = 0.8
EmbMatrix[Alice, 1] = 0.3
EmbMatrix[Alice, 2] = 0.5

// Bob's embedding: [0.7, 0.4, 0.6]
// Similar to Alice (family members often have similar patterns)
EmbMatrix[Bob, 0] = 0.7
EmbMatrix[Bob, 1] = 0.4
EmbMatrix[Bob, 2] = 0.6

// Charlie's embedding: [0.6, 0.5, 0.7]
// Similar to Bob
EmbMatrix[Charlie, 0] = 0.6
EmbMatrix[Charlie, 1] = 0.5
EmbMatrix[Charlie, 2] = 0.7

// Dave's embedding: [0.2, 0.8, 0.1]
// Very different from the family cluster
EmbMatrix[Dave, 0] = 0.2
EmbMatrix[Dave, 1] = 0.8
EmbMatrix[Dave, 2] = 0.1

// Eve's embedding: [0.75, 0.35, 0.55]
// Very similar to Alice (maybe relatives?)
EmbMatrix[Eve, 0] = 0.75
EmbMatrix[Eve, 1] = 0.35
EmbMatrix[Eve, 2] = 0.55

// Entity embedding lookup
// Emb[entity, d] retrieves the d-th dimension of entity's embedding
Emb[entity, d] = EmbMatrix[entity, d]

// ============================================
// Symbolic Component: Knowledge Base (Facts)
// ============================================
// Known parent-child relationships
// These are ground truths from a database or knowledge graph

Parent(Alice, Bob)
Parent(Bob, Charlie)

// Note: We DON'T have explicit facts for:
// - Parent(Eve, Alice) - but we might infer they're related
// - Parent(Dave, anyone) - he's not in this family

// ============================================
// Compute Entity Similarities (Neural)
// ============================================
// Compute cosine similarity between all entity pairs
// Similarity = dot product of normalized embeddings
//
// For simplicity, using dot product directly
// (assumes embeddings are already normalized)
//
// Similarity score between entity x and entity y:
// sim(x, y) = Emb[x, 0]*Emb[y, 0] + Emb[x, 1]*Emb[y, 1] + Emb[x, 2]*Emb[y, 2]

SimilarityScore[x, y] = Emb[x, d] Emb[y, d]

// Expected similarity scores:
// sim(Alice, Bob) = 0.8*0.7 + 0.3*0.4 + 0.5*0.6
//                 = 0.56 + 0.12 + 0.30 = 0.98 (high - parent/child)
//
// sim(Alice, Eve) = 0.8*0.75 + 0.3*0.35 + 0.5*0.55
//                 = 0.60 + 0.105 + 0.275 = 0.98 (high - possibly related!)
//
// sim(Alice, Dave) = 0.8*0.2 + 0.3*0.8 + 0.5*0.1
//                  = 0.16 + 0.24 + 0.05 = 0.45 (low - unrelated)
//
// sim(Bob, Charlie) = 0.7*0.6 + 0.4*0.5 + 0.6*0.7
//                   = 0.42 + 0.20 + 0.42 = 1.04 (high - parent/child)

// ============================================
// Define Similarity Threshold
// ============================================
// Entities are considered "similar" if their dot product exceeds this
threshold = 0.8

// ============================================
// Hybrid Rule: Similarity-Based Reasoning
// ============================================
// Derive Similar(x, y) relation from embeddings
// Two entities are similar if their embedding similarity > threshold
//
// This is a HYBRID rule: uses neural similarity + symbolic logic
//
// For each pair (x, y), check if:
// SimilarityScore[x, y] > threshold
//
// Note: We compute this explicitly rather than in rule body
// to work around grammar limitations with comparisons in assignments

// Create boolean tensor for similarity
// Similar[x, y] = 1 if SimilarityScore[x, y] > threshold, else 0

Similar[Alice, Bob] = step(SimilarityScore[Alice, Bob] - threshold)
Similar[Alice, Charlie] = step(SimilarityScore[Alice, Charlie] - threshold)
Similar[Alice, Dave] = step(SimilarityScore[Alice, Dave] - threshold)
Similar[Alice, Eve] = step(SimilarityScore[Alice, Eve] - threshold)
Similar[Bob, Alice] = step(SimilarityScore[Bob, Alice] - threshold)
Similar[Bob, Charlie] = step(SimilarityScore[Bob, Charlie] - threshold)
Similar[Bob, Dave] = step(SimilarityScore[Bob, Dave] - threshold)
Similar[Bob, Eve] = step(SimilarityScore[Bob, Eve] - threshold)
Similar[Charlie, Alice] = step(SimilarityScore[Charlie, Alice] - threshold)
Similar[Charlie, Bob] = step(SimilarityScore[Charlie, Bob] - threshold)
Similar[Charlie, Dave] = step(SimilarityScore[Charlie, Dave] - threshold)
Similar[Charlie, Eve] = step(SimilarityScore[Charlie, Eve] - threshold)
Similar[Dave, Alice] = step(SimilarityScore[Dave, Alice] - threshold)
Similar[Dave, Bob] = step(SimilarityScore[Dave, Bob] - threshold)
Similar[Dave, Charlie] = step(SimilarityScore[Dave, Charlie] - threshold)
Similar[Dave, Eve] = step(SimilarityScore[Dave, Eve] - threshold)
Similar[Eve, Alice] = step(SimilarityScore[Eve, Alice] - threshold)
Similar[Eve, Bob] = step(SimilarityScore[Eve, Bob] - threshold)
Similar[Eve, Charlie] = step(SimilarityScore[Eve, Charlie] - threshold)
Similar[Eve, Dave] = step(SimilarityScore[Eve, Dave] - threshold)

// Alternative: Using Datalog-style fact generation
// (This would be the ideal syntax once comparison literals in rules are fully supported)
// Similar(x, y) <- SimilarityScore[x, y] > threshold

// ============================================
// Symbolic Reasoning: Ancestor Relationship
// ============================================
// Standard transitive closure for family relationships

Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// This derives:
// Ancestor(Alice, Bob) - from Parent
// Ancestor(Bob, Charlie) - from Parent
// Ancestor(Alice, Charlie) - from transitivity

// ============================================
// Neurosymbolic Inference: Discover Potential Relations
// ============================================
// Use BOTH similarity (neural) AND family structure (symbolic)
// to infer potential undocumented relationships
//
// Rule: If x is similar to y, and y is parent of z,
//       then x MIGHT BE RELATED to z
//
// This discovers relationships not in the knowledge base
// by combining learned patterns with logical rules

MaybeRelated(x, z) <- Similar(x, y), Parent(y, z), x != z

// Expected inferences:
//
// MaybeRelated(Eve, Bob):
//   - Similar(Eve, Alice) is true (similarity score 0.98 > 0.8)
//   - Parent(Alice, Bob) is true
//   - Therefore: Eve might be related to Bob
//   - Interpretation: Eve could be Alice's sister, making her Bob's aunt
//
// MaybeRelated(Alice, Charlie):
//   - Similar(Alice, Bob) is true (similarity score 0.98 > 0.8)
//   - Parent(Bob, Charlie) is true
//   - Therefore: Alice might be related to Charlie
//   - Interpretation: This is actually TRUE - Alice is Charlie's grandparent!
//
// MaybeRelated(Eve, Charlie):
//   - Similar(Eve, Alice) is true OR Similar(Eve, Bob) is true
//   - Following parent chains
//   - Therefore: Eve might be related to Charlie

// Additional rule: Potential family cluster
// If two people are similar, they might be in the same family
PotentialFamily(x, y) <- Similar(x, y), x != y

// ============================================
// Queries: Explore Neural-Symbolic Inferences
// ============================================

// Query 1: What is the similarity between Alice and Eve?
SimilarityScore[Alice, Eve]?
// Expected: ~0.98 (very high - suggests relationship)

// Query 2: What is the similarity between Alice and Dave?
SimilarityScore[Alice, Dave]?
// Expected: ~0.45 (low - unrelated)

// Query 3: Who is similar to Alice?
Similar[Alice, y]?
// Expected: Bob, Eve (both have similarity > 0.8)

// Query 4: Who is similar to Dave?
Similar[Dave, y]?
// Expected: Possibly none (Dave is an outlier)

// Query 5: What are the known parent relationships?
Parent(x, y)?
// Expected: (Alice, Bob), (Bob, Charlie)

// Query 6: What are the derived ancestor relationships?
Ancestor(x, y)?
// Expected: (Alice, Bob), (Bob, Charlie), (Alice, Charlie)

// Query 7: Who might Eve be related to? (Neurosymbolic inference!)
MaybeRelated(Eve, z)?
// Expected: Bob, Charlie
// Reasoning: Eve is similar to Alice, and Alice's descendants are Bob and Charlie

// Query 8: Who might be related to Charlie?
MaybeRelated(x, Charlie)?
// Expected: Alice (from Similar(Alice, Bob) + Parent(Bob, Charlie))
//           Eve (from Similar(Eve, Alice) + ancestor chain)

// Query 9: Potential family clusters
PotentialFamily(x, y)?
// Expected: All pairs with high similarity
//           (Alice, Bob), (Alice, Eve), (Bob, Charlie), etc.

// ============================================
// Verification Queries
// ============================================

// Check specific similarity values
SimilarityScore[Alice, Bob]?   // Should be ~0.98
SimilarityScore[Bob, Charlie]? // Should be ~1.04
SimilarityScore[Alice, Eve]?   // Should be ~0.98
SimilarityScore[Alice, Dave]?  // Should be ~0.45

// Check similarity boolean
Similar[Alice, Eve]?  // Should be 1 (True)
Similar[Alice, Dave]? // Should be 0 (False)

// Check ancestor derivations
Ancestor(Alice, Charlie)? // Should be True (derived)
Ancestor(Charlie, Alice)? // Should be False (wrong direction)

// Check neurosymbolic inferences
MaybeRelated(Eve, Bob)?     // Should be True
MaybeRelated(Dave, Bob)?    // Should be False (Dave not similar to anyone)
MaybeRelated(Alice, Charlie)? // Should be True (actually is ancestor!)