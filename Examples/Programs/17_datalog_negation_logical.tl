Person(Alice)
Person(Bob)
Person(Charlie)

// Need to explicitly declare both for symmetry
Friend(Alice, Bob)
Friend(Bob, Alice)

// NonFriend(x,y) holds when both are persons, distinct, and not friends
NonFriend(x, y) <- Person(x), Person(y), x != y, ¬Friend(x, y)

NonFriend(x, y)?
