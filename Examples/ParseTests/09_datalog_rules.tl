// Example 9: Datalog Rules
// Transitive closure for ancestor relationship

// Base case: parent is ancestor
Ancestor(x, y) <- Parent(x, y)

// Recursive case: ancestor of parent is ancestor
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)
