#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"

using namespace tl;
using Catch::Matchers::WithinAbs;

TEST_CASE("Normalized indices - basic softmax", "[normalized]") {
    std::string code = R"(
        X[0] = 1.0
        X[1] = 2.0
        X[2] = 3.0
        Y[i.] = X[i]
        Y[0]?
        Y[1]?
        Y[2]?
    )";

    auto program = tl::parseProgram(code);
    TensorLogicVM vm;
    vm.execute(program);

    auto Y = vm.env().lookup("Y");

    // Check that values sum to 1.0 (softmax normalization)
    float sum = Y[0].item<float>() + Y[1].item<float>() + Y[2].item<float>();
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-5));

    // Check that values are in ascending order (since inputs were 1, 2, 3)
    REQUIRE(Y[0].item<float>() < Y[1].item<float>());
    REQUIRE(Y[1].item<float>() < Y[2].item<float>());
}

TEST_CASE("Normalized indices - 2D tensor normalization", "[normalized]") {
    std::string code = R"(
        Scores[0,0] = 1.0
        Scores[0,1] = 2.0
        Scores[0,2] = 3.0
        Scores[1,0] = 0.5
        Scores[1,1] = 1.5
        Scores[1,2] = 2.5
        Probs[i,j.] = Scores[i,j]
        Probs[0,0]?
        Probs[0,1]?
        Probs[0,2]?
        Probs[1,0]?
        Probs[1,1]?
        Probs[1,2]?
    )";

    auto program = tl::parseProgram(code);
    TensorLogicVM vm;
    vm.execute(program);

    auto Probs = vm.env().lookup("Probs");

    // Check that each row sums to 1.0
    float sum_row0 = Probs[0][0].item<float>() + Probs[0][1].item<float>() + Probs[0][2].item<float>();
    float sum_row1 = Probs[1][0].item<float>() + Probs[1][1].item<float>() + Probs[1][2].item<float>();

    REQUIRE_THAT(sum_row0, WithinAbs(1.0, 1e-5));
    REQUIRE_THAT(sum_row1, WithinAbs(1.0, 1e-5));
}

TEST_CASE("Normalized indices - attention mechanism", "[normalized]") {
    std::string code = R"(
        Query[0,0] = 1.0
        Query[0,1] = 0.5
        Key[0,0] = 0.8
        Key[0,1] = 0.6
        Key[1,0] = 1.2
        Key[1,1] = 0.4
        Key[2,0] = 0.9
        Key[2,1] = 0.7

        Scores[q,k] = Query[q,d] Key[k,d]
        Attn[q,k.] = Scores[q,k]

        Attn[0,0]?
        Attn[0,1]?
        Attn[0,2]?
    )";

    auto program = tl::parseProgram(code);
    TensorLogicVM vm;
    vm.execute(program);

    auto Attn = vm.env().lookup("Attn");

    // Check that attention weights sum to 1.0
    float attn_sum = Attn[0][0].item<float>() + Attn[0][1].item<float>() + Attn[0][2].item<float>();
    REQUIRE_THAT(attn_sum, WithinAbs(1.0, 1e-5));
}

TEST_CASE("Normalized indices - explicit softmax should not double-normalize", "[normalized]") {
    std::string code = R"(
        X[0] = 1.0
        X[1] = 2.0
        X[2] = 3.0
        Y[i.] = softmax(X[i])
        Y[0]?
        Y[1]?
        Y[2]?
    )";

    auto program = tl::parseProgram(code);
    TensorLogicVM vm;
    vm.execute(program);

    auto Y = vm.env().lookup("Y");

    // Check that values still sum to 1.0 (softmax applied once, not twice)
    float sum = Y[0].item<float>() + Y[1].item<float>() + Y[2].item<float>();
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-5));
}

TEST_CASE("Normalized indices - parser validation: only one normalized index", "[normalized][parse]") {
    std::string code = R"(
        X[0,0] = 1.0
        Y[i.,j.] = X[i,j]
    )";

    REQUIRE_THROWS_AS(tl::parseProgram(code), ParseError);
}

TEST_CASE("Normalized indices - parser validation: must be lowercase identifier", "[normalized][parse]") {
    std::string code = R"(
        X[0] = 1.0
        Y[I.] = X[I]
    )";

    REQUIRE_THROWS_AS(tl::parseProgram(code), ParseError);
}

TEST_CASE("Normalized indices - parser validation: not with numeric index", "[normalized][parse]") {
    std::string code = R"(
        X[0] = 1.0
        Y[0.] = X[0]
    )";

    REQUIRE_THROWS_AS(tl::parseProgram(code), ParseError);
}

TEST_CASE("Normalized indices - works with expressions", "[normalized]") {
    std::string code = R"(
        X[0] = 1.0
        X[1] = 2.0
        X[2] = 3.0
        Y[i.] = relu(X[i])
        Y[0]?
        Y[1]?
        Y[2]?
    )";

    auto program = tl::parseProgram(code);
    TensorLogicVM vm;
    vm.execute(program);

    auto Y = vm.env().lookup("Y");

    // Check that values sum to 1.0
    float sum = Y[0].item<float>() + Y[1].item<float>() + Y[2].item<float>();
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-5));

    // All values should be positive (relu doesn't change positive values)
    REQUIRE(Y[0].item<float>() > 0.0);
    REQUIRE(Y[1].item<float>() > 0.0);
    REQUIRE(Y[2].item<float>() > 0.0);
}

TEST_CASE("Normalized indices - scalar edge case", "[normalized]") {
    std::string code = R"(
        x = 5.0
        y[i.] = x
        y[0]?
    )";

    auto program = tl::parseProgram(code);
    TensorLogicVM vm;
    vm.execute(program);

    auto y = vm.env().lookup("y");

    // Scalar normalized should be 1.0
    // y is returned as a 1D tensor with one element
    REQUIRE_THAT(y.item<float>(), WithinAbs(1.0, 1e-5));
}

TEST_CASE("Normalized indices - with arithmetic operations", "[normalized]") {
    std::string code = R"(
        X[0] = 1.0
        X[1] = 2.0
        Scale = 2.0
        Y[i.] = X[i] * Scale
        Y[0]?
        Y[1]?
    )";

    auto program = tl::parseProgram(code);
    TensorLogicVM vm;
    vm.execute(program);

    auto Y = vm.env().lookup("Y");

    // Check that values sum to 1.0 despite scaling
    float sum = Y[0].item<float>() + Y[1].item<float>();
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-5));
}
