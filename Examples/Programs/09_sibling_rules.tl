// ============================================
// Sibling Relationship Rules
// ============================================
// Demonstrates rules with multiple conditions

// Parent facts
Parent(Alice, Bob)
Parent(Alice, Carol)
Parent(Alice, Dave)
Parent(Eve, Frank)
Parent(Eve, Grace)

// Rule: Two people are siblings if they share a parent
// x and y are siblings if there exists a person p who is parent of both
// Note: This includes x being sibling of x (can be refined)
Sibling(x, y) <- Parent(p, x), Parent(p, y)

// ============================================
// Derived Sibling Facts
// ============================================
// After forward chaining:
//   Sibling(Bob, Bob)     [reflexive - same parent Alice]
//   Sibling(Bob, Carol)   [share parent Alice]
//   Sibling(Bob, Dave)    [share parent Alice]
//   Sibling(Carol, Bob)   [share parent Alice]
//   Sibling(Carol, Carol) [reflexive]
//   Sibling(Carol, Dave)  [share parent Alice]
//   Sibling(Dave, Bob)    [share parent Alice]
//   Sibling(Dave, Carol)  [share parent Alice]
//   Sibling(Dave, Dave)   [reflexive]
//   Sibling(Frank, Frank) [reflexive]
//   Sibling(Frank, Grace) [share parent Eve]
//   Sibling(Grace, Frank) [share parent Eve]
//   Sibling(Grace, Grace) [reflexive]

// Queries
Sibling(Bob, Carol)?
// Expected: True

Sibling(Bob, Frank)?
// Expected: False (different families)

Sibling(Bob, x)?
// Expected: Bob, Carol, Dave