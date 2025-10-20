// ============================================
// Facts Across Multiple Domains
// ============================================

// People and their attributes
Person(Alice)
Person(Bob)
Person(Charlie)

// Gender facts
Male(Bob)
Male(Charlie)
Female(Alice)

// Age facts (person, age)
Age(Alice, 45)
Age(Bob, 25)
Age(Charlie, 5)

// Location facts (person, city)
LivesIn(Alice, Seattle)
LivesIn(Bob, Portland)
LivesIn(Charlie, Portland)

// Occupation facts
Occupation(Alice, Engineer)
Occupation(Bob, Teacher)

// ============================================
// Queries Across Domains
// ============================================

// Who are the males?
Male(person)?
// Expected: Bob, Charlie

// Who lives in Portland?
LivesIn(person, Portland)?
// Expected: Bob, Charlie

// What is Bob's occupation?
Occupation(Bob, job)?
// Expected: Teacher

// What is Alice's age?
Age(Alice, age)?
// Expected: 45

// Who is a female engineer?
// (This would require a rule to combine facts - see next example)
Female(person)?
Occupation(person, Engineer)?
// Expected: Alice