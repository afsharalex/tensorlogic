#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <cmath>
#include <sstream>

using namespace tl;
using Catch::Matchers::WithinAbs;

// Helper to get scalar value from tensor
static float getScalar(const torch::Tensor& t) {
    return t.item<float>();
}

// Helper to get tensor value at indices
static float getTensorValue(const torch::Tensor& t, const std::vector<int64_t>& indices) {
    torch::Tensor indexed = t;
    for (auto idx : indices) {
        indexed = indexed.index({idx});
    }
    return indexed.item<float>();
}

TEST_CASE("Basic scalar assignment", "[tensor][scalar]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram("x = 42");
    vm.execute(prog);

    auto x = vm.env().lookup("x");
    REQUIRE_THAT(getScalar(x), WithinAbs(42.0f, 0.001f));
}

TEST_CASE("Element-wise tensor assignment", "[tensor][element]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        W[0, 0] = 1.0
        W[0, 1] = 2.0
        W[1, 0] = 3.0
        W[1, 1] = 4.0
    )");
    vm.execute(prog);

    auto W = vm.env().lookup("W");
    REQUIRE_THAT(getTensorValue(W, {0, 0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(W, {0, 1}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(W, {1, 0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(W, {1, 1}), WithinAbs(4.0f, 0.001f));
}

TEST_CASE("List literal 1D initialization", "[tensor][list]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram("V = [1, 2, 3, 4]");
    vm.execute(prog);

    auto V = vm.env().lookup("V");
    REQUIRE(V.dim() == 1);
    REQUIRE(V.size(0) == 4);
    REQUIRE_THAT(getTensorValue(V, {0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(V, {1}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(V, {2}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(V, {3}), WithinAbs(4.0f, 0.001f));
}

TEST_CASE("List literal 2D initialization", "[tensor][list]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram("M = [[1, 2], [3, 4]]");
    vm.execute(prog);

    auto M = vm.env().lookup("M");
    REQUIRE(M.dim() == 2);
    REQUIRE(M.size(0) == 2);
    REQUIRE(M.size(1) == 2);
    REQUIRE_THAT(getTensorValue(M, {0, 0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(M, {0, 1}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(M, {1, 0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(M, {1, 1}), WithinAbs(4.0f, 0.001f));
}

TEST_CASE("Vector dot product (einsum)", "[tensor][einsum]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        A = [1, 2, 3]
        B = [4, 5, 6]
        C = A[i] B[i]
    )");
    vm.execute(prog);

    // C should be 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    auto C = vm.env().lookup("C");
    REQUIRE_THAT(getScalar(C), WithinAbs(32.0f, 0.001f));
}

TEST_CASE("Matrix-vector multiply", "[tensor][einsum]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        W = [[1, 2], [3, 4]]
        X = [10, 20]
        Y[i] = W[i, j] X[j]
    )");
    vm.execute(prog);

    // Y[0] = 1*10 + 2*20 = 10 + 40 = 50
    // Y[1] = 3*10 + 4*20 = 30 + 80 = 110
    auto Y = vm.env().lookup("Y");
    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(50.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(110.0f, 0.001f));
}

TEST_CASE("Matrix-matrix multiply", "[tensor][einsum]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C[i, k] = A[i, j] B[j, k]
    )");
    vm.execute(prog);

    // C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
    // C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
    // C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
    // C[1,1] = 3*6 + 4*8 = 18 + 32 = 50
    auto C = vm.env().lookup("C");
    REQUIRE_THAT(getTensorValue(C, {0, 0}), WithinAbs(19.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(C, {0, 1}), WithinAbs(22.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(C, {1, 0}), WithinAbs(43.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(C, {1, 1}), WithinAbs(50.0f, 0.001f));
}

TEST_CASE("Sum reduction", "[tensor][reduction]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        V = [1, 2, 3, 4]
        total = V[i]
    )");
    vm.execute(prog);

    // total should be 1 + 2 + 3 + 4 = 10
    auto total = vm.env().lookup("total");
    REQUIRE_THAT(getScalar(total), WithinAbs(10.0f, 0.001f));
}

TEST_CASE("Arithmetic addition", "[tensor][arithmetic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        A = [1, 2, 3]
        B = [4, 5, 6]
        C[i] = A[i] + B[i]
    )");
    vm.execute(prog);

    auto C = vm.env().lookup("C");
    REQUIRE_THAT(getTensorValue(C, {0}), WithinAbs(5.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(C, {1}), WithinAbs(7.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(C, {2}), WithinAbs(9.0f, 0.001f));
}

TEST_CASE("Arithmetic subtraction", "[tensor][arithmetic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        A = [10, 20, 30]
        B = [1, 2, 3]
        C[i] = A[i] - B[i]
    )");
    vm.execute(prog);

    auto C = vm.env().lookup("C");
    REQUIRE_THAT(getTensorValue(C, {0}), WithinAbs(9.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(C, {1}), WithinAbs(18.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(C, {2}), WithinAbs(27.0f, 0.001f));
}

TEST_CASE("Scalar-vector multiplication", "[tensor][arithmetic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        scale = 2.0
        V = [1, 2, 3]
        W[i] = scale V[i]
    )");
    vm.execute(prog);

    auto W = vm.env().lookup("W");
    REQUIRE_THAT(getTensorValue(W, {0}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(W, {1}), WithinAbs(4.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(W, {2}), WithinAbs(6.0f, 0.001f));
}

TEST_CASE("Identity assignment", "[tensor][identity]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        X = [[1, 2], [3, 4]]
        Y = X
    )");
    vm.execute(prog);

    auto Y = vm.env().lookup("Y");
    // Identity assignment should copy the full tensor
    REQUIRE(Y.dim() == 2);
    REQUIRE(Y.size(0) == 2);
    REQUIRE(Y.size(1) == 2);
    REQUIRE_THAT(getTensorValue(Y, {0, 0}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {0, 1}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1, 0}), WithinAbs(3.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Y, {1, 1}), WithinAbs(4.0f, 0.001f));
}

TEST_CASE("Label-based indexing", "[tensor][labels]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        W[Alice] = 1.0
        W[Bob] = 2.0
        W[Charlie] = 3.0
    )");
    vm.execute(prog);

    auto W = vm.env().lookup("W");
    // Labels should be mapped to indices 0, 1, 2
    REQUIRE(W.dim() == 1);
    REQUIRE(W.size(0) >= 3);
}

TEST_CASE("Scalar exponentiation", "[tensor][exponentiation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram("result = 2^3");
    vm.execute(prog);

    auto result = vm.env().lookup("result");
    REQUIRE_THAT(getScalar(result), WithinAbs(8.0f, 0.001f));
}

TEST_CASE("Right-associative exponentiation", "[tensor][exponentiation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    // 2^(3^2) = 2^9 = 512 (right-associative)
    // NOT (2^3)^2 = 8^2 = 64 (left-associative)
    auto prog = parseProgram("result = 2^3^2");
    vm.execute(prog);

    auto result = vm.env().lookup("result");
    REQUIRE_THAT(getScalar(result), WithinAbs(512.0f, 0.001f));
}

TEST_CASE("Element-wise tensor exponentiation", "[tensor][exponentiation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        Base = [2, 3, 4]
        Exp = [2, 2, 2]
        Result[i] = Base[i]^Exp[i]
    )");
    vm.execute(prog);

    // Result should be [4, 9, 16]
    auto Result = vm.env().lookup("Result");
    REQUIRE_THAT(getTensorValue(Result, {0}), WithinAbs(4.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Result, {1}), WithinAbs(9.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Result, {2}), WithinAbs(16.0f, 0.001f));
}

TEST_CASE("Exponentiation with scalar and tensor", "[tensor][exponentiation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        V = [1, 2, 3]
        Result[i] = 2^V[i]
    )");
    vm.execute(prog);

    // Result should be [2^1, 2^2, 2^3] = [2, 4, 8]
    auto Result = vm.env().lookup("Result");
    REQUIRE_THAT(getTensorValue(Result, {0}), WithinAbs(2.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Result, {1}), WithinAbs(4.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Result, {2}), WithinAbs(8.0f, 0.001f));
}

TEST_CASE("Polynomial expression with exponentiation", "[tensor][exponentiation]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    // Compute y = x^2 + 2*x + 1 for x = 3
    auto prog = parseProgram(R"(
        x = 3
        y = x^2 + 2*x + 1
    )");
    vm.execute(prog);

    // y = 3^2 + 2*3 + 1 = 9 + 6 + 1 = 16
    auto y = vm.env().lookup("y");
    REQUIRE_THAT(getScalar(y), WithinAbs(16.0f, 0.001f));
}
