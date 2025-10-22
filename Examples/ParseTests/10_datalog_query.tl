// Example 10: Datalog Query
// Define facts and rules, then query

Parent(Alice, Bob)
Parent(Bob, Charlie)

Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// Query: Who are the ancestors of Charlie?
Ancestor(x, Charlie)?
