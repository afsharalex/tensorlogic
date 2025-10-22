// ============================================
// Facts with Temporal/Contextual Information
// ============================================
// Using ternary relations to add context

// Parent relationships with birth year
// Format: Parent(parent, child, year)
Parent(Alice, Bob, 1990)
Parent(Bob, Charlie, 2015)
Parent(Charlie, David, 2040)

// Query: Who was born in 2015?
Parent(p, c, 2015)?
// Expected: p=Bob, c=Charlie

// Query: What year was Charlie born?
Parent(Bob, Charlie, year)?
// Expected: year=2015

// Query: All parent-child pairs and their years
Parent(p, c, y)?