#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <sstream>

using namespace tl;
using Catch::Matchers::WithinAbs;

// Helper to get tensor value at indices
static float getTensorValue(const torch::Tensor& t, const std::vector<int64_t>& indices) {
    torch::Tensor indexed = t;
    for (auto idx : indices) {
        indexed = indexed.index({idx});
    }
    return indexed.item<float>();
}

TEST_CASE("Slice with start:end (1D)", "[slice][basic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5]
        Y = X[1:4]
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // Y should be [2, 3, 4] (indices 1, 2, 3)
    REQUIRE(Y.dim() == 1);
    REQUIRE(Y.size(0) == 3);
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(4.0f, 0.001f));
}

TEST_CASE("Slice with :end (from beginning)", "[slice][basic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5]
        Y = X[:3]
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // Y should be [1, 2, 3] (indices 0, 1, 2)
    REQUIRE(Y.dim() == 1);
    REQUIRE(Y.size(0) == 3);
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(3.0f, 0.001f));
}

TEST_CASE("Slice with start: (to end)", "[slice][basic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5]
        Y = X[2:]
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // Y should be [3, 4, 5] (indices 2, 3, 4)
    REQUIRE(Y.dim() == 1);
    REQUIRE(Y.size(0) == 3);
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(4.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(5.0f, 0.001f));
}

TEST_CASE("Slice with : (all elements)", "[slice][basic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5]
        Y = X[:]
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // Y should be [1, 2, 3, 4, 5] (all elements)
    REQUIRE(Y.dim() == 1);
    REQUIRE(Y.size(0) == 5);
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(4.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {4}), WithinAbs(5.0f, 0.001f));
}

TEST_CASE("Slice with start:end:step", "[slice][step]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5, 6, 7, 8]
        Y = X[0:8:2]
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // Y should be [1, 3, 5, 7] (every second element)
    REQUIRE(Y.dim() == 1);
    REQUIRE(Y.size(0) == 4);
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(5.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(7.0f, 0.001f));
}

TEST_CASE("Slice 2D tensor with mixed indices and slices", "[slice][2d]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        M = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        Row = M[1, :]
    )");
    vm.execute(prog);

    auto Row = vm.env().lookup("Row");
    // Row should be [5, 6, 7, 8] (second row)
    REQUIRE(Row.dim() == 1);
    REQUIRE(Row.size(0) == 4);
    REQUIRE_THAT(getTensorValue(Row, {0}), WithinAbs(5.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Row, {1}), WithinAbs(6.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Row, {2}), WithinAbs(7.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Row, {3}), WithinAbs(8.0f, 0.001f));
}

TEST_CASE("Slice 2D tensor with slice:slice", "[slice][2d]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        M = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        Sub = M[0:2, 1:3]
    )");
    vm.execute(prog);

    auto Sub = vm.env().lookup("Sub");
    // Sub should be [[2, 3], [6, 7]]
    REQUIRE(Sub.dim() == 2);
    REQUIRE(Sub.size(0) == 2);
    REQUIRE(Sub.size(1) == 2);
    REQUIRE_THAT(getTensorValue(Sub, {0, 0}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Sub, {0, 1}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Sub, {1, 0}), WithinAbs(6.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Sub, {1, 1}), WithinAbs(7.0f, 0.001f));
}

TEST_CASE("Slice 2D tensor column selection", "[slice][2d]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        M = [[1, 2, 3], [4, 5, 6]]
        Col = M[:, 1]
    )");
    vm.execute(prog);

    auto Col = vm.env().lookup("Col");
    // Col should be [2, 5] (second column)
    REQUIRE(Col.dim() == 1);
    REQUIRE(Col.size(0) == 2);
    REQUIRE_THAT(getTensorValue(Col, {0}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Col, {1}), WithinAbs(5.0f, 0.001f));
}

TEST_CASE("Slice assignment (write to slice)", "[slice][assign]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5]
        X[1:4] = [10, 20, 30]
    )");
    vm.execute(prog);

    auto X = vm.env().lookup("X");
    // X should be [1, 10, 20, 30, 5]
    REQUIRE_THAT(getTensorValue(X, {0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(X, {1}), WithinAbs(10.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(X, {2}), WithinAbs(20.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(X, {3}), WithinAbs(30.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(X, {4}), WithinAbs(5.0f, 0.001f));
}

// TODO: Chained indexing like X[1:4][i] requires parser extension
// TEST_CASE("Slice in tensor equation (RHS)", "[slice][equation][!mayfail]") {
//     std::stringstream out, err;
//     TensorLogicVM vm{&out, &err};
//     auto prog = parseProgram(R"(
//         X = [1, 2, 3, 4, 5]
//         Y[i] = 2.0 * X[1:4][i]
//     )");
//     vm.execute(prog);
//
//     auto Y = vm.env().lookup("Y");
//     // Y should be [4, 6, 8] (2 * [2, 3, 4])
//     REQUIRE(Y.dim() == 1);
//     REQUIRE(Y.size(0) == 3);
//     REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(4.0f, 0.001f));
//     REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(6.0f, 0.001f));
//     REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(8.0f, 0.001f));
// }

TEST_CASE("Negative indices in slicing", "[slice][negative]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [1, 2, 3, 4, 5]
        Y = X[-3:-1]
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // Y should be [3, 4] (last 3 elements minus last element)
    REQUIRE(Y.dim() == 1);
    REQUIRE(Y.size(0) == 2);
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(4.0f, 0.001f));
}
