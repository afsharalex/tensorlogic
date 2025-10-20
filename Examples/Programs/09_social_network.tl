// ============================================
// Social Network - Friend Relationships
// ============================================

// Direct friendships (symmetric - both directions)
Friend(Alice, Bob)
Friend(Bob, Alice)
Friend(Bob, Carol)
Friend(Carol, Bob)
Friend(Carol, Dave)
Friend(Dave, Carol)
Friend(Dave, Eve)
Friend(Eve, Dave)

// Rule: You know someone directly if they're your friend
Knows(x, y) <- Friend(x, y)

// Rule: You know friend-of-friend (up to 2 hops)
Knows(x, z) <- Friend(x, y), Friend(y, z)

// ============================================
// Derived Knowledge
// ============================================
// Direct:
//   All Friend relationships become Knows relationships
//
// Friend-of-friend:
//   Knows(Alice, Carol) [Alice -> Bob -> Carol]
//   Knows(Bob, Dave)    [Bob -> Carol -> Dave]
//   Knows(Carol, Eve)   [Carol -> Dave -> Eve]
//   (and reverses)

// Queries
Knows(Alice, Carol)?
// Expected: True (friend of friend)

Knows(Alice, Dave)?
// Expected: False (3 hops away, rule only goes 2 hops)

Knows(Alice, x)?
// Expected: Bob, Carol, Alice (direct friend and friend-of-friend)