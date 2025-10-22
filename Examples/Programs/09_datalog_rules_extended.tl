// ============================================
// Extended Family Tree with Multiple Branches
// ============================================
//
// Family structure:
//
//        George
//       /      \
//    John      Sarah
//     |          |
//   Tom        Mike
//     |
//   Jane

// Define parent relationships
Parent(George, John)
Parent(George, Sarah)
Parent(John, Tom)
Parent(Sarah, Mike)
Parent(Tom, Jane)

// Define ancestor rules (same as before)
Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// ============================================
// Derived Ancestor Facts
// ============================================
// After forward chaining, these will be derived:
//
// Direct (1 generation):
//   Ancestor(George, John)
//   Ancestor(George, Sarah)
//   Ancestor(John, Tom)
//   Ancestor(Sarah, Mike)
//   Ancestor(Tom, Jane)
//
// 2 generations:
//   Ancestor(George, Tom)    [George -> John -> Tom]
//   Ancestor(George, Mike)   [George -> Sarah -> Mike]
//   Ancestor(John, Jane)     [John -> Tom -> Jane]
//
// 3 generations:
//   Ancestor(George, Jane)   [George -> John -> Tom -> Jane]

// ============================================
// Queries
// ============================================

// Is George an ancestor of Jane?
Ancestor(George, Jane)?
// Expected: True (3 generations removed)

// Who are George's descendants?
Ancestor(George, y)?
// Expected: John, Sarah, Tom, Mike, Jane

// Who are Tom's ancestors?
Ancestor(x, Tom)?
// Expected: John, George

// Is Mike an ancestor of John?
Ancestor(Mike, John)?
// Expected: False (they're cousins, not in ancestor relationship)