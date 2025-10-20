// ============================================
// Datalog Query Example
// ============================================
// This program demonstrates the complete workflow:
// 1. Define base facts (knowledge base)
// 2. Define inference rules (logic)
// 3. Execute queries (ask questions)
//
// The system uses forward chaining to compute all derivable
// facts, then answers queries by looking up the results.

// ============================================
// Define Base Facts: Parent Relationships
// ============================================
// These are the ground truths we start with
//
// Simple family tree:
//   Alice
//     |
//    Bob
//     |
//  Charlie

Parent(Alice, Bob)
Parent(Bob, Charlie)

// ============================================
// Define Inference Rules: Ancestor Relationships
// ============================================
// Rules allow us to infer new facts from existing ones

// Rule 1: Base case
// Direct parents are ancestors
Ancestor(x, y) <- Parent(x, y)

// Rule 2: Recursive case
// Transitive closure: if x is ancestor of y, and y is parent of z,
// then x is ancestor of z
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// ============================================
// Forward Chaining Process
// ============================================
// The system automatically derives all ancestor facts:
//
// Iteration 1 (Apply base case rule):
//   From Parent(Alice, Bob) -> Ancestor(Alice, Bob)
//   From Parent(Bob, Charlie) -> Ancestor(Bob, Charlie)
//
// Iteration 2 (Apply recursive rule):
//   From Ancestor(Alice, Bob) + Parent(Bob, Charlie)
//     -> Ancestor(Alice, Charlie)
//
// Iteration 3:
//   No new facts can be derived -> Fixpoint reached
//
// Final knowledge base contains:
//   Parent(Alice, Bob)
//   Parent(Bob, Charlie)
//   Ancestor(Alice, Bob)
//   Ancestor(Bob, Charlie)
//   Ancestor(Alice, Charlie)

// ============================================
// Queries
// ============================================

// Query 1: Who are the ancestors of Charlie?
// This finds all x where Ancestor(x, Charlie) is true
Ancestor(x, Charlie)?
// Expected results: Alice, Bob
// Explanation:
//   - Bob is Charlie's parent (direct ancestor)
//   - Alice is Charlie's grandparent (indirect ancestor via Bob)

// Query 2: Who are Bob's ancestors?
Ancestor(x, Bob)?
// Expected result: Alice

// Query 3: Is Alice an ancestor of Charlie?
// This is a ground query (no variables)
Ancestor(Alice, Charlie)?
// Expected result: True

// Query 4: Is Charlie an ancestor of Alice?
// Testing the reverse relationship
Ancestor(Charlie, Alice)?
// Expected result: False
// Explanation: Ancestor relationship is not symmetric

// Query 5: Who are Alice's descendants?
// (People that Alice is an ancestor of)
Ancestor(Alice, y)?
// Expected results: Bob, Charlie

// Query 6: All ancestor relationships
// Find all pairs (x, y) where x is ancestor of y
Ancestor(x, y)?
// Expected results:
//   (Alice, Bob)
//   (Alice, Charlie)
//   (Bob, Charlie)

// Query 7: All parent relationships (original facts)
Parent(x, y)?
// Expected results:
//   (Alice, Bob)
//   (Bob, Charlie)