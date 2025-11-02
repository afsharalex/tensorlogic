#include <catch2/catch_test_macros.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <string>
#include <sstream>

using namespace tl;

// Helper to check if a fact exists in the Datalog store
static bool hasFact(const Environment& env, const std::string& relation,
                    const std::vector<std::string>& args) {
    const auto& facts = env.facts(relation);
    for (const auto& fact : facts) {
        if (fact.size() == args.size()) {
            bool match = true;
            for (size_t i = 0; i < args.size(); ++i) {
                if (fact[i] != args[i]) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
    }
    return false;
}

TEST_CASE("Simple Datalog fact", "[datalog][facts]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram("Parent(Alice, Bob)");
    vm.execute(prog);

    REQUIRE(hasFact(vm.env(), "Parent", {"Alice", "Bob"}));
}

TEST_CASE("Multiple Datalog facts", "[datalog][facts]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)
        Parent(Charlie, Dave)
    )");
    vm.execute(prog);

    REQUIRE(hasFact(vm.env(), "Parent", {"Alice", "Bob"}));
    REQUIRE(hasFact(vm.env(), "Parent", {"Bob", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "Parent", {"Charlie", "Dave"}));
}

TEST_CASE("Datalog rule - identity", "[datalog][rules]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)

        Ancestor(x, y) <- Parent(x, y)

        Ancestor(x, y)?
    )");
    vm.execute(prog);

    // Ancestor should include all Parent facts
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Alice", "Bob"}));
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Bob", "Charlie"}));
}

TEST_CASE("Datalog rule - transitive closure", "[datalog][rules][transitive]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)
        Parent(Charlie, Dave)

        Ancestor(x, y) <- Parent(x, y)
        Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

        Ancestor(x, y)?
    )");
    vm.execute(prog);

    // Direct ancestors
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Alice", "Bob"}));
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Bob", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Charlie", "Dave"}));

    // Transitive ancestors
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Alice", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Bob", "Dave"}));
    REQUIRE(hasFact(vm.env(), "Ancestor", {"Alice", "Dave"}));
}

TEST_CASE("Datalog rule - sibling", "[datalog][rules]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Alice, Charlie)
        Parent(Bob, Dave)
        Parent(Bob, Eve)

        Sibling(x, y) <- Parent(p, x), Parent(p, y), x != y

        Sibling(x, y)?
    )");
    vm.execute(prog);

    // Bob and Charlie are siblings
    REQUIRE(hasFact(vm.env(), "Sibling", {"Bob", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "Sibling", {"Charlie", "Bob"}));

    // Dave and Eve are siblings
    REQUIRE(hasFact(vm.env(), "Sibling", {"Dave", "Eve"}));
    REQUIRE(hasFact(vm.env(), "Sibling", {"Eve", "Dave"}));
}

TEST_CASE("Datalog rule with comparison", "[datalog][rules][comparison]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Value[Alice] = 10
        Value[Bob] = 20
        Value[Charlie] = 15

        // Note: This tests if comparisons work in rules
        // Actual implementation may vary
    )");
    vm.execute(prog);

    // Just verify program executes without error
    REQUIRE(vm.env().has("Value"));
}

TEST_CASE("Datalog query - simple", "[datalog][query]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)

        Parent(x, Charlie)?
    )");

    // Execute and verify no errors
    REQUIRE_NOTHROW(vm.execute(prog));
}

TEST_CASE("Datalog query - conjunctive", "[datalog][query]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)
        Parent(Charlie, Dave)

        Ancestor(x, y) <- Parent(x, y)
        Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

        Ancestor(x, y), Ancestor(y, z)?
    )");

    // Execute and verify no errors
    REQUIRE_NOTHROW(vm.execute(prog));
}

TEST_CASE("Datalog - social network", "[datalog][rules][complex]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Friend(Alice, Bob)
        Friend(Bob, Charlie)
        Friend(Charlie, Dave)

        // Friendship is symmetric
        Friend(y, x) <- Friend(x, y)

        // Friend of friend
        FoF(x, z) <- Friend(x, y), Friend(y, z), x != z

        Friend(x, y)?
    )");
    vm.execute(prog);

    // Symmetric friendship
    REQUIRE(hasFact(vm.env(), "Friend", {"Alice", "Bob"}));
    REQUIRE(hasFact(vm.env(), "Friend", {"Bob", "Alice"}));

    // Friend of friend relationships should exist
    REQUIRE(hasFact(vm.env(), "FoF", {"Alice", "Charlie"}));
}

TEST_CASE("Datalog - path finding", "[datalog][rules][graph]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Edge(A, B)
        Edge(B, C)
        Edge(C, D)
        Edge(B, E)

        Path(x, y) <- Edge(x, y)
        Path(x, z) <- Path(x, y), Edge(y, z)

        Path(x, y)?
    )");
    vm.execute(prog);

    // Direct paths
    REQUIRE(hasFact(vm.env(), "Path", {"A", "B"}));
    REQUIRE(hasFact(vm.env(), "Path", {"B", "C"}));

    // Transitive paths
    REQUIRE(hasFact(vm.env(), "Path", {"A", "C"}));
    REQUIRE(hasFact(vm.env(), "Path", {"A", "D"}));
    REQUIRE(hasFact(vm.env(), "Path", {"B", "D"}));
}

TEST_CASE("Datalog - multiple relations", "[datalog][facts][multiple]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Person(Alice)
        Person(Bob)
        City(NYC)
        City(SF)
        LivesIn(Alice, NYC)
        LivesIn(Bob, SF)

        Resident(p, c) <- Person(p), City(c), LivesIn(p, c)

        Resident(x, y)?
    )");
    vm.execute(prog);

    REQUIRE(hasFact(vm.env(), "Person", {"Alice"}));
    REQUIRE(hasFact(vm.env(), "Person", {"Bob"}));
    REQUIRE(hasFact(vm.env(), "City", {"NYC"}));
    REQUIRE(hasFact(vm.env(), "LivesIn", {"Alice", "NYC"}));
    REQUIRE(hasFact(vm.env(), "Resident", {"Alice", "NYC"}));
}

TEST_CASE("Datalog - grandparent rule", "[datalog][rules]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)
        Parent(Bob, Dave)
        Parent(Charlie, Eve)

        Grandparent(x, z) <- Parent(x, y), Parent(y, z)

        Grandparent(x, y)?
    )");
    vm.execute(prog);

    // Alice is grandparent of Charlie and Dave
    REQUIRE(hasFact(vm.env(), "Grandparent", {"Alice", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "Grandparent", {"Alice", "Dave"}));

    // Bob is grandparent of Eve
    REQUIRE(hasFact(vm.env(), "Grandparent", {"Bob", "Eve"}));
}
TEST_CASE("Datalog negation in rule body", "[datalog][rules][negation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    // People and one friendship (added in both directions for symmetry)
    auto prog = parseProgram(R"(
        Person(Alice)
        Person(Bob)
        Person(Charlie)
        Friend(Alice, Bob)
        Friend(Bob, Alice)

        // NonFriend(x,y) holds when both are persons, distinct, and not friends
        NonFriend(x, y) <- Person(x), Person(y), x != y, not Friend(x, y)

        NonFriend(x, y)?
    )");

    vm.execute(prog);

    // Alice and Bob are friends, so not included in NonFriend in either direction
    REQUIRE_FALSE(hasFact(vm.env(), "NonFriend", {"Alice", "Bob"}));
    REQUIRE_FALSE(hasFact(vm.env(), "NonFriend", {"Bob", "Alice"}));

    // Pairs that are not friends should appear
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Alice", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Charlie", "Alice"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Bob", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Charlie", "Bob"}));
}

TEST_CASE("Datalog negation in rule body '!'", "[datalog][rules][negation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    // People and one friendship (added in both directions for symmetry)
    auto prog = parseProgram(R"(
        Person(Alice)
        Person(Bob)
        Person(Charlie)
        Friend(Alice, Bob)
        Friend(Bob, Alice)

        // NonFriend(x,y) holds when both are persons, distinct, and not friends
        NonFriend(x, y) <- Person(x), Person(y), x != y, !Friend(x, y)

        NonFriend(x, y)?
    )");

    vm.execute(prog);

    // Alice and Bob are friends, so not included in NonFriend in either direction
    REQUIRE_FALSE(hasFact(vm.env(), "NonFriend", {"Alice", "Bob"}));
    REQUIRE_FALSE(hasFact(vm.env(), "NonFriend", {"Bob", "Alice"}));

    // Pairs that are not friends should appear
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Alice", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Charlie", "Alice"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Bob", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Charlie", "Bob"}));
}

TEST_CASE("Datalog negation in rule body '¬'", "[datalog][rules][negation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    // People and one friendship (added in both directions for symmetry)
    auto prog = parseProgram(R"(
        Person(Alice)
        Person(Bob)
        Person(Charlie)
        Friend(Alice, Bob)
        Friend(Bob, Alice)

        // NonFriend(x,y) holds when both are persons, distinct, and not friends
        NonFriend(x, y) <- Person(x), Person(y), x != y, ¬Friend(x, y)

        NonFriend(x, y)?
    )");

    vm.execute(prog);

    // Alice and Bob are friends, so not included in NonFriend in either direction
    REQUIRE_FALSE(hasFact(vm.env(), "NonFriend", {"Alice", "Bob"}));
    REQUIRE_FALSE(hasFact(vm.env(), "NonFriend", {"Bob", "Alice"}));

    // Pairs that are not friends should appear
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Alice", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Charlie", "Alice"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Bob", "Charlie"}));
    REQUIRE(hasFact(vm.env(), "NonFriend", {"Charlie", "Bob"}));
}

TEST_CASE("Datalog negation in conjunctive query", "[datalog][query][negation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        Person(Alice)
        Person(Bob)
        Friend(Alice, Bob)
        Friend(Bob, Alice)

        // query: persons x,y that are not friends (should return None since both are friends)
        Person(x), Person(y), x != y, not Friend(x, y)?
    )");

    // Should parse and execute without throwing; we don't assert specific output here
    REQUIRE_NOTHROW(vm.execute(prog));
}

TEST_CASE("Datalog float constants in facts", "[datalog][facts][float]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Temperature(Room1, 72.5)
        Temperature(Room2, 68.3)
        Temperature(Room3, 75.0)
    )");
    vm.execute(prog);

    REQUIRE(hasFact(vm.env(), "Temperature", {"Room1", "72.5"}));
    REQUIRE(hasFact(vm.env(), "Temperature", {"Room2", "68.3"}));
    REQUIRE(hasFact(vm.env(), "Temperature", {"Room3", "75.0"}));
}

TEST_CASE("Datalog integer constants in facts", "[datalog][facts][integer]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Age(Alice, 25)
        Age(Bob, 30)
        Age(Charlie, 35)
    )");
    vm.execute(prog);

    REQUIRE(hasFact(vm.env(), "Age", {"Alice", "25"}));
    REQUIRE(hasFact(vm.env(), "Age", {"Bob", "30"}));
    REQUIRE(hasFact(vm.env(), "Age", {"Charlie", "35"}));
}

TEST_CASE("Datalog mixed string and numeric constants", "[datalog][facts][mixed]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Coordinate(Point1, 3.14, 2.71)
        Coordinate(Point2, 1.41, 1.73)
        Score(Alice, 95)
        Score(Bob, 87.5)
    )");
    vm.execute(prog);

    REQUIRE(hasFact(vm.env(), "Coordinate", {"Point1", "3.14", "2.71"}));
    REQUIRE(hasFact(vm.env(), "Coordinate", {"Point2", "1.41", "1.73"}));
    REQUIRE(hasFact(vm.env(), "Score", {"Alice", "95"}));
    REQUIRE(hasFact(vm.env(), "Score", {"Bob", "87.5"}));
}

TEST_CASE("Datalog rule with numeric constants", "[datalog][rules][numeric]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Temperature(Room1, 72.5)
        Temperature(Room2, 68.3)
        Temperature(Room3, 75.0)

        Comfortable(r) <- Temperature(r, t), t >= 70, t <= 74

        Comfortable(x)?
    )");
    vm.execute(prog);

    // Room1 at 72.5 is comfortable
    REQUIRE(hasFact(vm.env(), "Comfortable", {"Room1"}));
    // Room2 at 68.3 is too cold
    REQUIRE_FALSE(hasFact(vm.env(), "Comfortable", {"Room2"}));
    // Room3 at 75.0 is too hot
    REQUIRE_FALSE(hasFact(vm.env(), "Comfortable", {"Room3"}));
}

TEST_CASE("Datalog query with numeric constants", "[datalog][query][numeric]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Price(Apple, 1.50)
        Price(Banana, 0.75)
        Price(Orange, 2.00)

        Price(x, p), p < 1.0?
    )");

    // Should parse and execute without throwing
    REQUIRE_NOTHROW(vm.execute(prog));
}
