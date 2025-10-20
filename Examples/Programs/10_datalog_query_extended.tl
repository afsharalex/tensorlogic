// ============================================
// Extended Datalog Query Example
// ============================================
// Larger family tree with more interesting queries

// ============================================
// Base Facts
// ============================================
//
// Family structure:
//
//      George --- Martha
//         |          |
//      +--+----------+--+
//      |                |
//    John            Sarah
//      |                |
//    Tom              Mike

// Generation 1: Grandparents
Parent(George, John)
Parent(Martha, John)
Parent(George, Sarah)
Parent(Martha, Sarah)

// Generation 2: Parents
Parent(John, Tom)
Parent(Sarah, Mike)

// ============================================
// Inference Rules
// ============================================

// Ancestor relationship (as before)
Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// Sibling relationship: share at least one parent
Sibling(x, y) <- Parent(p, x), Parent(p, y), x != y

// Grandparent relationship: parent of parent
Grandparent(x, z) <- Parent(x, y), Parent(y, z)

// Cousin relationship: parents are siblings
Cousin(x, y) <- Parent(px, x), Parent(py, y), Sibling(px, py), x != y

// ============================================
// Queries
// ============================================

// Who are Tom's ancestors?
Ancestor(x, Tom)?
// Expected: John, George, Martha

// Who are George's descendants?
Ancestor(George, y)?
// Expected: John, Sarah, Tom, Mike

// Who are John's siblings?
Sibling(John, x)?
// Expected: Sarah

// Who are Tom's grandparents?
Grandparent(x, Tom)?
// Expected: George, Martha

// Who are Tom's cousins?
Cousin(Tom, x)?
// Expected: Mike

// Are George and Tom directly related by Parent?
Parent(George, Tom)?
// Expected: False (George is Tom's grandparent, not parent)

// Are George and Tom related by Ancestor?
Ancestor(George, Tom)?
// Expected: True