#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"

using namespace tl;
using Catch::Matchers::WithinAbs;

static float getTensorValue(const torch::Tensor& t, const std::vector<int64_t>& indices) {
    torch::Tensor indexed = t;
    for (auto idx : indices) {
        indexed = indexed.index({idx});
    }
    return indexed.item<float>();
}

TEST_CASE("1D max pooling with stride 2", "[pooling][max]") {
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        X = [1, 5, 3, 7, 2, 8, 4, 6]
        Y[i/2] max= X[i]
    )");
    vm.execute(prog);

    // Y[0] = max(1, 5) = 5
    // Y[1] = max(3, 7) = 7
    // Y[2] = max(2, 8) = 8
    // Y[3] = max(4, 6) = 6
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(5.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(7.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(8.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(6.0f, 0.001f));
}

TEST_CASE("1D average pooling with stride 2", "[pooling][avg]") {
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        X = [1, 5, 3, 7, 2, 8, 4, 6]
        Y[i/2] avg= X[i]
    )");
    vm.execute(prog);

    // Y[0] = avg(1, 5) = 3.0
    // Y[1] = avg(3, 7) = 5.0
    // Y[2] = avg(2, 8) = 5.0
    // Y[3] = avg(4, 6) = 5.0
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(5.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(5.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(5.0f, 0.001f));
}

TEST_CASE("1D min pooling with stride 2", "[pooling][min]") {
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        X = [1, 5, 3, 7, 2, 8, 4, 6]
        Y[i/2] min= X[i]
    )");
    vm.execute(prog);

    // Y[0] = min(1, 5) = 1
    // Y[1] = min(3, 7) = 3
    // Y[2] = min(2, 8) = 2
    // Y[3] = min(4, 6) = 4
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(4.0f, 0.001f));
}

TEST_CASE("1D sum reduction with stride 2", "[pooling][sum]") {
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5, 6]
        Y[i/2] += X[i]
    )");
    vm.execute(prog);

    // Y[0] = 1 + 2 = 3
    // Y[1] = 3 + 4 = 7
    // Y[2] = 5 + 6 = 11
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(7.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(11.0f, 0.001f));
}

TEST_CASE("2D max pooling with stride 2", "[pooling][max][2d]") {
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        // 4x4 image
        Image[0, 0] = 1.0
        Image[0, 1] = 2.0
        Image[0, 2] = 3.0
        Image[0, 3] = 4.0
        Image[1, 0] = 5.0
        Image[1, 1] = 6.0
        Image[1, 2] = 7.0
        Image[1, 3] = 8.0
        Image[2, 0] = 9.0
        Image[2, 1] = 8.0
        Image[2, 2] = 7.0
        Image[2, 3] = 6.0
        Image[3, 0] = 5.0
        Image[3, 1] = 4.0
        Image[3, 2] = 3.0
        Image[3, 3] = 2.0

        // 2x2 max pooling with stride 2
        Pooled[x/2, y/2] max= Image[x, y]
    )");
    vm.execute(prog);

    // Pooled[0, 0] = max(1, 2, 5, 6) = 6
    // Pooled[0, 1] = max(3, 4, 7, 8) = 8
    // Pooled[1, 0] = max(9, 8, 5, 4) = 9
    // Pooled[1, 1] = max(7, 6, 3, 2) = 7
    auto Pooled = vm.env().lookup("Pooled");
    REQUIRE_THAT(getTensorValue(Pooled, {0, 0}), WithinAbs(6.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Pooled, {0, 1}), WithinAbs(8.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Pooled, {1, 0}), WithinAbs(9.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Pooled, {1, 1}), WithinAbs(7.0f, 0.001f));
}

TEST_CASE("2D average pooling with stride 2", "[pooling][avg][2d]") {
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        // 4x4 image
        Image[0, 0] = 1.0
        Image[0, 1] = 2.0
        Image[0, 2] = 3.0
        Image[0, 3] = 4.0
        Image[1, 0] = 5.0
        Image[1, 1] = 6.0
        Image[1, 2] = 7.0
        Image[1, 3] = 8.0
        Image[2, 0] = 9.0
        Image[2, 1] = 8.0
        Image[2, 2] = 7.0
        Image[2, 3] = 6.0
        Image[3, 0] = 5.0
        Image[3, 1] = 4.0
        Image[3, 2] = 3.0
        Image[3, 3] = 2.0

        // 2x2 average pooling with stride 2
        Pooled[x/2, y/2] avg= Image[x, y]
    )");
    vm.execute(prog);

    // Pooled[0, 0] = avg(1, 2, 5, 6) = 14/4 = 3.5
    // Pooled[0, 1] = avg(3, 4, 7, 8) = 22/4 = 5.5
    // Pooled[1, 0] = avg(9, 8, 5, 4) = 26/4 = 6.5
    // Pooled[1, 1] = avg(7, 6, 3, 2) = 18/4 = 4.5
    auto Pooled = vm.env().lookup("Pooled");
    REQUIRE_THAT(getTensorValue(Pooled, {0, 0}), WithinAbs(3.5f, 0.001f));
    REQUIRE_THAT(getTensorValue(Pooled, {0, 1}), WithinAbs(5.5f, 0.001f));
    REQUIRE_THAT(getTensorValue(Pooled, {1, 0}), WithinAbs(6.5f, 0.001f));
    REQUIRE_THAT(getTensorValue(Pooled, {1, 1}), WithinAbs(4.5f, 0.001f));
}

TEST_CASE("Pooling with stride 3", "[pooling][stride]") {
    TensorLogicVM vm;
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        Y[i/3] max= X[i]
    )");
    vm.execute(prog);

    // Y[0] = max(1, 2, 3) = 3
    // Y[1] = max(4, 5, 6) = 6
    // Y[2] = max(7, 8, 9) = 9
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(6.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(9.0f, 0.001f));
}
