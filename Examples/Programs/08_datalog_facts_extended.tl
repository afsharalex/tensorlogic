// ============================================
// Extended Family Relationships
// ============================================
// A more realistic family tree with multiple branches
//
// Family structure:
//
//     George --- Martha
//        |          |
//     +--+----------+--+
//     |                |
//   John            Sarah
//     |                |
//   +---+            +---+
//   |   |            |   |
// Tom  Ann        Mike  Lisa

// Generation 1: Grandparents
Parent(George, John)
Parent(Martha, John)
Parent(George, Sarah)
Parent(Martha, Sarah)

// Generation 2: Parents
Parent(John, Tom)
Parent(John, Ann)
Parent(Sarah, Mike)
Parent(Sarah, Lisa)

// ============================================
// Queries
// ============================================

// Who are John's parents?
Parent(p, John)?
// Expected: George, Martha

// Who are John's children?
Parent(John, c)?
// Expected: Tom, Ann

// Is George a parent of Tom?
Parent(George, Tom)?
// Expected: False (George is Tom's grandparent, not direct parent)

// Who are all of Sarah's children?
Parent(Sarah, child)?
// Expected: Mike, Lisa

// List all parent relationships
Parent(x, y)?