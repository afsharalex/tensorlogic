// ============================================
// Complex Query Example
// ============================================
// Finding specific relationship patterns

// Facts
Parent(Alice, Bob)
Parent(Bob, Charlie)
Parent(Charlie, David)
Parent(Alice, Eve)
Parent(Eve, Frank)

// Rules
Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z), x != z

// Descendant is the inverse of ancestor
Descendant(x, y) <- Ancestor(y, x)

// Common ancestor: someone who is ancestor to both
CommonAncestor(a, x, y) <- Ancestor(a, x), Ancestor(a, y), x != y

// ============================================
// Complex Queries
// ============================================

// Who is a common ancestor of Charlie and Frank?
CommonAncestor(a, Charlie, Frank)?
// Expected: Alice (ancestor of both through different paths)

// Find all pairs of people who share a common ancestor
CommonAncestor(a, x, y)?
// Expected: Multiple triples showing all shared ancestry

// How many generations separate Alice from David?
// (This would require counting, which is advanced)
// For now, we can just verify the relationship exists:
Ancestor(Alice, David)?
// Expected: True (Alice -> Bob -> Charlie -> David)