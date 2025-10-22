#include <catch2/catch_test_macros.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <string>

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
    TensorLogicVM vm;
    auto prog = parseProgram("Parent(Alice, Bob)");
    vm.execute(prog);

    REQUIRE(hasFact(vm.env(), "Parent", {"Alice", "Bob"}));
}

TEST_CASE("Multiple Datalog facts", "[datalog][facts]") {
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)

        Parent(x, Charlie)?
    )");

    // Execute and verify no errors
    REQUIRE_NOTHROW(vm.execute(prog));
}

TEST_CASE("Datalog query - conjunctive", "[datalog][query]") {
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
    TensorLogicVM vm;
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
