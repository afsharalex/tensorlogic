// ============================================
// Multiple Relationship Types
// ============================================
// Demonstrating different kinds of relationships

// Parent relationships
Parent(Alice, Bob)
Parent(Alice, Carol)
Parent(Bob, David)

// Sibling relationships
Sibling(Bob, Carol)
Sibling(Carol, Bob)

// Spouse relationships
Spouse(Alice, Edward)
Spouse(Edward, Alice)

// Friend relationships
Friend(Bob, Frank)
Friend(Frank, Bob)
Friend(Carol, Grace)

// ============================================
// Queries on Different Relations
// ============================================

// Who are Alice's children?
Parent(Alice, child)?
// Expected: Bob, Carol

// Who are Bob's siblings?
Sibling(Bob, sib)?
// Expected: Carol

// Who is Alice's spouse?
Spouse(Alice, partner)?
// Expected: Edward

// Who are Bob's friends?
Friend(Bob, friend)?
// Expected: Frank

// Is Carol a sibling of Bob?
Sibling(Carol, Bob)?
// Expected: True

// Is David a sibling of Bob?
Sibling(David, Bob)?
// Expected: False (David is Bob's child)