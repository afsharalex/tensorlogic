#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <cmath>
#include <sstream>

using namespace tl;
using Catch::Matchers::WithinAbs;

static float getScalar(const torch::Tensor& t) {
    return t.item<float>();
}

static float getTensorValue(const torch::Tensor& t, const std::vector<int64_t>& indices) {
    torch::Tensor indexed = t;
    for (auto idx : indices) {
        indexed = indexed.index({idx});
    }
    return indexed.item<float>();
}

TEST_CASE("ReLU activation", "[activation][relu]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        X = [-2, -1, 0, 1, 2]
        Y[i] = relu(X[i])
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(0.0f, 0.001f)); // relu(-2) = 0
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(0.0f, 0.001f)); // relu(-1) = 0
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(0.0f, 0.001f)); // relu(0) = 0
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(1.0f, 0.001f)); // relu(1) = 1
    REQUIRE_THAT(getTensorValue(Y, {4}), WithinAbs(2.0f, 0.001f)); // relu(2) = 2
}

TEST_CASE("Sigmoid activation", "[activation][sigmoid]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        X = [0, 1, -1]
        Y[i] = sigmoid(X[i])
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // sigmoid(0) = 0.5
    // sigmoid(1) ≈ 0.731
    // sigmoid(-1) ≈ 0.269
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(0.5f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(0.731f, 0.01f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(0.269f, 0.01f));
}

TEST_CASE("Tanh activation", "[activation][tanh]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        X = [0, 1, -1]
        Y[i] = tanh(X[i])
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // tanh(0) = 0
    // tanh(1) ≈ 0.762
    // tanh(-1) ≈ -0.762
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(0.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(0.762f, 0.01f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(-0.762f, 0.01f));
}

TEST_CASE("Step (Heaviside) function", "[activation][step]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        X = [-2, -0.5, 0, 0.5, 2]
        Y[i] = step(X[i])
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(0.0f, 0.001f)); // step(-2) = 0
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(0.0f, 0.001f)); // step(-0.5) = 0
    // step(0) can be 0 or 1 depending on implementation
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(1.0f, 0.001f)); // step(0.5) = 1
    REQUIRE_THAT(getTensorValue(Y, {4}), WithinAbs(1.0f, 0.001f)); // step(2) = 1
}

TEST_CASE("Exponential function", "[activation][exp]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        X = [0, 1, 2]
        Y[i] = exp(X[i])
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // exp(0) = 1.0
    // exp(1) = e ≈ 2.718
    // exp(2) = e^2 ≈ 7.389
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(2.718f, 0.01f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(7.389f, 0.01f));
}

TEST_CASE("Square root function", "[activation][sqrt]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        X = [0, 1, 4, 9, 16]
        Y[i] = sqrt(X[i])
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(0.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {4}), WithinAbs(4.0f, 0.001f));
}

TEST_CASE("Absolute value function", "[activation][abs]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        X = [-2, -1, 0, 1, 2]
        Y[i] = abs(X[i])
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(0.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {4}), WithinAbs(2.0f, 0.001f));
}

TEST_CASE("Single layer perceptron with sigmoid", "[activation][perceptron]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        // Inputs
        X = [1, 0]

        // Weights
        W = [0.5, 0.3]

        // Bias
        b = -0.2

        // Compute weighted sum
        z = W[j] X[j] + b

        // Apply activation
        Y = sigmoid(z)
    )");
    vm.execute(prog);

    // z = 0.5*1 + 0.3*0 - 0.2 = 0.3
    // Y = sigmoid(0.3) ≈ 0.574
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getScalar(Y), WithinAbs(0.574f, 0.01f));
}

TEST_CASE("Multi-layer perceptron", "[activation][mlp]") {
    std::stringstream out, err;
    TensorLogicVM vm(&out, &err);
    auto prog = parseProgram(R"(
        // Input
        X = [1, 2]

        // First layer weights (2x3)
        W1 = [[0.5, 0.3, 0.2], [0.4, 0.6, 0.1]]

        // First layer bias
        b1 = [0.1, 0.2, 0.3]

        // Hidden layer activation
        H[j] = relu(W1[i, j] X[i] + b1[j])

        // Second layer weights (3x1)
        W2 = [0.5, 0.3, 0.2]

        // Output
        Y = sigmoid(W2[j] H[j])
    )");
    vm.execute(prog);

    // H[0] = relu(0.5*1 + 0.4*2 + 0.1) = relu(1.6) = 1.6
    // H[1] = relu(0.3*1 + 0.6*2 + 0.2) = relu(1.7) = 1.7
    // H[2] = relu(0.2*1 + 0.1*2 + 0.3) = relu(0.7) = 0.7
    // Y = sigmoid(0.5*1.6 + 0.3*1.7 + 0.2*0.7) = sigmoid(1.45) ≈ 0.810
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getScalar(Y), WithinAbs(0.810f, 0.02f));
}
