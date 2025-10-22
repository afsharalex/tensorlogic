// ============================================
// Queries with Constraints
// ============================================

// Facts
Parent(Alice, Bob)
Parent(Alice, Carol)
Parent(Alice, Dave)
Parent(Bob, Eve)

Age(Alice, 50)
Age(Bob, 25)
Age(Carol, 22)
Age(Dave, 20)
Age(Eve, 5)

// Rules
Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// Adult children: descendants who are adults (age >= 18)
AdultDescendant(x, y) <- Ancestor(x, y), Age(y, a), a >= 18, x != y

// Younger sibling: siblings where one is younger than the other
YoungerSibling(x, y) <- Parent(p, x), Parent(p, y),
                        Age(x, ax), Age(y, ay), ax < ay, x != y

// ============================================
// Queries with Constraints
// ============================================

// Who are Alice's adult descendants?
AdultDescendant(Alice, x)?
// Expected: Bob, Carol, Dave (not Eve, who is 5)

// Who are Bob's younger siblings?
YoungerSibling(x, Bob)?
// Expected: Carol, Dave (both younger than Bob's 25 years)

// Find all ancestor-descendant pairs where both are adults
AdultDescendant(x, y)?
// Expected: All pairs where ancestor and descendant are both >= 18