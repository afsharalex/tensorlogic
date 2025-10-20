// ============================================
// Datalog Rules - Ancestor Relationships
// ============================================
// This program defines rules for inferring ancestor relationships
// from parent facts using transitive closure
//
// The <- operator means "if" (implication)
// Commas on the right side mean "and" (conjunction)
//
// This demonstrates:
// - Base case rules (direct facts)
// - Recursive rules (inductive reasoning)
// - Transitive closure (computing multi-step relationships)

// ============================================
// Define Base Facts: Parent Relationships
// ============================================
// Starting knowledge base
//
// Family tree:
//        Eve
//         |
//       Alice
//         |
//        Bob
//         |
//      Charlie
//         |
//       David

Parent(Eve, Alice)
Parent(Alice, Bob)
Parent(Bob, Charlie)
Parent(Charlie, David)

// ============================================
// Define Ancestor Rules
// ============================================

// Rule 1: Base case
// If x is a parent of y, then x is an ancestor of y
// This establishes the foundation for ancestry
Ancestor(x, y) <- Parent(x, y)

// Rule 2: Recursive case (Transitive closure)
// If x is an ancestor of y, and y is a parent of z,
// then x is an ancestor of z
// This allows multi-generational ancestry inference
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// ============================================
// How the Rules Work
// ============================================
// Starting with Parent facts, forward chaining applies rules:
//
// Iteration 1 (Base case):
//   From Parent(Eve, Alice) -> Ancestor(Eve, Alice)
//   From Parent(Alice, Bob) -> Ancestor(Alice, Bob)
//   From Parent(Bob, Charlie) -> Ancestor(Bob, Charlie)
//   From Parent(Charlie, David) -> Ancestor(Charlie, David)
//
// Iteration 2 (First recursion):
//   From Ancestor(Eve, Alice) + Parent(Alice, Bob) -> Ancestor(Eve, Bob)
//   From Ancestor(Alice, Bob) + Parent(Bob, Charlie) -> Ancestor(Alice, Charlie)
//   From Ancestor(Bob, Charlie) + Parent(Charlie, David) -> Ancestor(Bob, David)
//
// Iteration 3 (Second recursion):
//   From Ancestor(Eve, Bob) + Parent(Bob, Charlie) -> Ancestor(Eve, Charlie)
//   From Ancestor(Alice, Charlie) + Parent(Charlie, David) -> Ancestor(Alice, David)
//
// Iteration 4 (Third recursion):
//   From Ancestor(Eve, Charlie) + Parent(Charlie, David) -> Ancestor(Eve, David)
//
// Iteration 5: No new facts can be derived -> fixpoint reached

// ============================================
// Expected Derived Facts (After Forward Chaining)
// ============================================
// Direct ancestors (1 generation):
//   Ancestor(Eve, Alice)
//   Ancestor(Alice, Bob)
//   Ancestor(Bob, Charlie)
//   Ancestor(Charlie, David)
//
// 2 generations:
//   Ancestor(Eve, Bob)
//   Ancestor(Alice, Charlie)
//   Ancestor(Bob, David)
//
// 3 generations:
//   Ancestor(Eve, Charlie)
//   Ancestor(Alice, David)
//
// 4 generations:
//   Ancestor(Eve, David)

// ============================================
// Queries
// ============================================

// Query 1: Is Alice an ancestor of Charlie?
Ancestor(Alice, Charlie)?
// Expected: True (Alice -> Bob -> Charlie)

// Query 2: Is Eve an ancestor of David?
Ancestor(Eve, David)?
// Expected: True (Eve -> Alice -> Bob -> Charlie -> David)

// Query 3: Is Bob an ancestor of Alice?
Ancestor(Bob, Alice)?
// Expected: False (relationship goes the other direction)

// Query 4: Who are Alice's ancestors?
Ancestor(x, Alice)?
// Expected: Eve

// Query 5: Who are Alice's descendants (people Alice is ancestor of)?
Ancestor(Alice, y)?
// Expected: Bob, Charlie, David

// Query 6: Who are Bob's descendants?
Ancestor(Bob, y)?
// Expected: Charlie, David

// Query 7: All ancestor relationships
Ancestor(x, y)?
// Expected: All 10 derived facts listed above