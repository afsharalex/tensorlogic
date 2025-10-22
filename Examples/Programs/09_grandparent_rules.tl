// ============================================
// Grandparent Relationship Rules
// ============================================

// Parent facts
Parent(George, John)
Parent(John, Tom)
Parent(John, Ann)
Parent(Sarah, Mike)

// Rule: x is grandparent of z if x is parent of y and y is parent of z
Grandparent(x, z) <- Parent(x, y), Parent(y, z)

// ============================================
// Derived Grandparent Facts
// ============================================
// Grandparent(George, Tom)
// Grandparent(George, Ann)

// Queries
Grandparent(George, Tom)?
// Expected: True

Grandparent(George, x)?
// Expected: Tom, Ann

Grandparent(x, Tom)?
// Expected: George