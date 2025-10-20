// ============================================
// Datalog Facts - Family Relationships
// ============================================
// This program defines a knowledge base of parent-child relationships
// using Datalog facts (Boolean relations)
//
// In Tensor Logic, Datalog facts use parentheses notation
// which indicates they are Boolean tensors (sparse representation)
//
// A fact Parent(Alice, Bob) means:
// - Alice is a parent of Bob
// - Represented as a Boolean tensor where Parent[Alice, Bob] = 1
// - All other combinations default to 0

// ============================================
// Define Parent Relationship Facts
// ============================================
// Format: Parent(parent_name, child_name)
//
// Family tree structure:
//
//        Eve
//         |
//       Alice
//         |
//        Bob
//         |
//      Charlie
//         |
//       David

// Eve is the parent of Alice
Parent(Eve, Alice)

// Alice is the parent of Bob
Parent(Alice, Bob)

// Bob is the parent of Charlie
Parent(Bob, Charlie)

// Charlie is the parent of David
Parent(Charlie, David)

// ============================================
// Query the Knowledge Base
// ============================================
// We can query specific relationships or all facts

// Query: Is Alice the parent of Bob?
Parent(Alice, Bob)?
// Expected: True (this fact exists)

// Query: Is Bob the parent of Alice?
Parent(Bob, Alice)?
// Expected: False (this fact does not exist - relationship is reversed)

// Query: Is Eve the parent of Alice?
Parent(Eve, Alice)?
// Expected: True

// Query: Is Alice the parent of David?
Parent(Alice, David)?
// Expected: False (Alice is not a direct parent of David)

// Query: Who are Bob's children?
Parent(Bob, x)?
// Expected: Charlie (x will be bound to Charlie)

// Query: Who are Alice's parents?
Parent(y, Alice)?
// Expected: Eve (y will be bound to Eve)

// Query: All parent-child relationships
Parent(x, y)?
// Expected: All four facts will be returned