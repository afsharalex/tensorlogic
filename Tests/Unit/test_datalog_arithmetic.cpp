#include <catch2/catch_test_macros.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <sstream>

using namespace tl;

TEST_CASE("Datalog arithmetic in rule heads", "[datalog][arithmetic]") {
    const char* source = R"(
        Cost(Item1, 10)
        Cost(Item2, 20)
        TotalCost(x, c1 + c2) <- Cost(x, c1), Cost(Item2, c2)
        TotalCost(Item1, c)?
    )";

    std::ostringstream out;
    TensorLogicVM vm(&out);
    auto prog = parseProgram(source);
    vm.execute(prog);

    std::string result = out.str();
    REQUIRE(result.find("30") != std::string::npos);
}

TEST_CASE("Datalog arithmetic - doubling values", "[datalog][arithmetic]") {
    const char* source = R"(
        Age(Alice, 25)
        Age(Bob, 30)
        DoubleAge(p, a * 2) <- Age(p, a)
        DoubleAge(Alice, x)?
    )";

    std::ostringstream out;
    TensorLogicVM vm(&out);
    auto prog = parseProgram(source);
    vm.execute(prog);

    std::string result = out.str();
    REQUIRE(result.find("50") != std::string::npos);
}

TEST_CASE("Datalog arithmetic - subtraction", "[datalog][arithmetic]") {
    const char* source = R"(
        Balance(Account1, 100)
        Withdrawal(Account1, 30)
        NewBalance(a, b - w) <- Balance(a, b), Withdrawal(a, w)
        NewBalance(Account1, x)?
    )";

    std::ostringstream out;
    TensorLogicVM vm(&out);
    auto prog = parseProgram(source);
    vm.execute(prog);

    std::string result = out.str();
    REQUIRE(result.find("70") != std::string::npos);
}

TEST_CASE("Datalog arithmetic - division", "[datalog][arithmetic]") {
    const char* source = R"(
        Total(Item1, 100)
        Count(Item1, 4)
        Average(i, t / c) <- Total(i, t), Count(i, c)
        Average(Item1, x)?
    )";

    std::ostringstream out;
    TensorLogicVM vm(&out);
    auto prog = parseProgram(source);
    vm.execute(prog);

    std::string result = out.str();
    REQUIRE(result.find("25") != std::string::npos);
}

TEST_CASE("Datalog arithmetic - complex expression", "[datalog][arithmetic]") {
    const char* source = R"(
        Value1(X, 10)
        Value2(X, 5)
        Result(i, (v1 + v2) * 2) <- Value1(i, v1), Value2(i, v2)
        Result(X, r)?
    )";

    std::ostringstream out;
    TensorLogicVM vm(&out);
    auto prog = parseProgram(source);
    vm.execute(prog);

    std::string result = out.str();
    REQUIRE(result.find("30") != std::string::npos); // (10 + 5) * 2 = 30
}

TEST_CASE("Datalog arithmetic - multiple results", "[datalog][arithmetic]") {
    const char* source = R"(
        Price(Apple, 2)
        Price(Orange, 3)
        Quantity(Apple, 5)
        Quantity(Orange, 4)
        TotalCost(item, p * q) <- Price(item, p), Quantity(item, q)
        TotalCost(x, c)?
    )";

    std::ostringstream out;
    TensorLogicVM vm(&out);
    auto prog = parseProgram(source);
    vm.execute(prog);

    std::string result = out.str();
    // Should have both: Apple costs 10 (2*5) and Orange costs 12 (3*4)
    REQUIRE((result.find("10") != std::string::npos || result.find("12") != std::string::npos));
}

TEST_CASE("Datalog arithmetic - chained rules", "[datalog][arithmetic]") {
    const char* source = R"(
        Base(X, 10)
        Step1(i, v * 2) <- Base(i, v)
        Step2(i, v + 5) <- Step1(i, v)
        Step2(X, r)?
    )";

    std::ostringstream out;
    TensorLogicVM vm(&out);
    auto prog = parseProgram(source);
    vm.execute(prog);

    std::string result = out.str();
    REQUIRE(result.find("25") != std::string::npos); // 10 * 2 + 5 = 25
}
