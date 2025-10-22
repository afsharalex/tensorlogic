// ============================================
// Explicit Boolean Tensor Representation
// ============================================
// This shows the equivalence between Datalog facts
// and Boolean tensor notation

// These two representations are equivalent:

// Datalog fact notation (compact, sparse)
Parent(Alice, Bob)

// Boolean tensor notation (explicit)
// Parent[Alice, Bob] = 1

// Unspecified combinations are implicitly 0:
// Parent[Alice, Alice] = 0 (implicit)
// Parent[Bob, Alice] = 0 (implicit)
// Parent[Charlie, Bob] = 0 (implicit)

// ============================================
// Small Complete Example
// ============================================

// Define facts using Datalog notation
Parent(A, B)
Parent(B, C)
Parent(B, D)

// These are equivalent to:
// Parent[A, B] = 1
// Parent[B, C] = 1
// Parent[B, D] = 1

// All other combinations are 0:
// Parent[A, A] = 0
// Parent[A, C] = 0
// Parent[A, D] = 0
// Parent[B, A] = 0
// Parent[B, B] = 0
// Parent[C, A] = 0
// etc.

// Query all facts
Parent(x, y)?
// Expected: (A, B), (B, C), (B, D)