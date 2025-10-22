// Example 14: Mixed Neurosymbolic Program
// Combines neural networks with symbolic reasoning

// Neural component: embedding lookup
Emb[entity, d] = EmbMatrix[entity, d]

// Symbolic component: knowledge base
Parent(Alice, Bob)
Parent(Bob, Charlie)

// Hybrid rule: similarity-based reasoning
Similar(x, y) <- Emb[x, d] Emb[y, d] > threshold

// Neural + symbolic: infer relationships
MaybeRelated(x, z) <- Similar(x, y), Parent(y, z)
