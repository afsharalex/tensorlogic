// ============================================
// Knowledge Graph Completion with Embeddings
// ============================================
// Complete missing edges in a knowledge graph
// using both logic and learned similarities

// Entity embeddings for knowledge graph entities
EmbMatrix[Paris, 0] = 0.9
EmbMatrix[Paris, 1] = 0.2
EmbMatrix[Paris, 2] = 0.3
EmbMatrix[France, 0] = 0.85
EmbMatrix[France, 1] = 0.25
EmbMatrix[France, 2] = 0.35
EmbMatrix[London, 0] = 0.88
EmbMatrix[London, 1] = 0.22
EmbMatrix[London, 2] = 0.32
EmbMatrix[UK, 0] = 0.82
EmbMatrix[UK, 1] = 0.28
EmbMatrix[UK, 2] = 0.38

Emb[entity, d] = EmbMatrix[entity, d]

// Known facts
CapitalOf(Paris, France)
CapitalOf(London, UK)

// Similarity
Similarity[x, y] = Emb[x, d] Emb[y, d]

// Predict missing relation:
// If x is capital of y, and x' is similar to x,
// and y' is similar to y,
// then x' might be capital of y'
PotentialCapital(xp, yp) <- CapitalOf(x, y),
                            Similarity[xp, x] > 0.9,
                            Similarity[yp, y] > 0.9,
                            xp != x

// Query: What might be capital of what?
PotentialCapital(x, y)?