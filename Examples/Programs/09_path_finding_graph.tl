// ============================================
// Path Finding in a Graph
// ============================================
// Similar structure to ancestor, but for general graphs
// Demonstrates transitive closure for reachability

// Define edges in a directed graph
// Graph structure:
//   A -> B -> D
//   |    |
//   v    v
//   C -> E

Edge(A, B)
Edge(A, C)
Edge(B, D)
Edge(B, E)
Edge(C, E)

// Rule: Base case - direct edge means reachable
Reachable(x, y) <- Edge(x, y)

// Rule: Recursive case - transitive reachability
Reachable(x, z) <- Reachable(x, y), Edge(y, z)

// ============================================
// Derived Reachability Facts
// ============================================
// Direct (1 hop):
//   Reachable(A, B), Reachable(A, C), Reachable(B, D),
//   Reachable(B, E), Reachable(C, E)
//
// 2 hops:
//   Reachable(A, D)  [A -> B -> D]
//   Reachable(A, E)  [A -> B -> E or A -> C -> E]

// Queries
Reachable(A, E)?
// Expected: True (multiple paths exist)

Reachable(A, x)?
// Expected: B, C, D, E (all nodes reachable from A)

Reachable(E, x)?
// Expected: None (E has no outgoing edges)