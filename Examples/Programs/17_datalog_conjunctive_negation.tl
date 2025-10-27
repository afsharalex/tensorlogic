Person(Alice)
Person(Bob)
Friend(Alice, Bob)
Friend(Bob, Alice)

// query: persons x,y that are not friends (should return None since both are friends)
Person(x), Person(y), x != y, not Friend(x, y)?
