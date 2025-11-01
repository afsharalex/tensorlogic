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

TEST_CASE("Tensor comparisons - scalar comparisons", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        a = 5.0
        b = 3.0
        c = 5.0

        lt_result = a < b
        le_result1 = a <= b
        le_result2 = a <= c
        gt_result = a > b
        ge_result1 = a >= b
        ge_result2 = a >= c
        eq_result1 = a == b
        eq_result2 = a == c
        ne_result1 = a != b
        ne_result2 = a != c
    )");

    vm.execute(prog);

    // Less than
    REQUIRE(vm.env().has("lt_result"));
    REQUIRE_THAT(vm.env().lookup("lt_result").item<float>(), WithinAbs(0.0f, 1e-5f));

    // Less than or equal
    REQUIRE(vm.env().has("le_result1"));
    REQUIRE_THAT(vm.env().lookup("le_result1").item<float>(), WithinAbs(0.0f, 1e-5f));
    REQUIRE(vm.env().has("le_result2"));
    REQUIRE_THAT(vm.env().lookup("le_result2").item<float>(), WithinAbs(1.0f, 1e-5f));

    // Greater than
    REQUIRE(vm.env().has("gt_result"));
    REQUIRE_THAT(vm.env().lookup("gt_result").item<float>(), WithinAbs(1.0f, 1e-5f));

    // Greater than or equal
    REQUIRE(vm.env().has("ge_result1"));
    REQUIRE_THAT(vm.env().lookup("ge_result1").item<float>(), WithinAbs(1.0f, 1e-5f));
    REQUIRE(vm.env().has("ge_result2"));
    REQUIRE_THAT(vm.env().lookup("ge_result2").item<float>(), WithinAbs(1.0f, 1e-5f));

    // Equal
    REQUIRE(vm.env().has("eq_result1"));
    REQUIRE_THAT(vm.env().lookup("eq_result1").item<float>(), WithinAbs(0.0f, 1e-5f));
    REQUIRE(vm.env().has("eq_result2"));
    REQUIRE_THAT(vm.env().lookup("eq_result2").item<float>(), WithinAbs(1.0f, 1e-5f));

    // Not equal
    REQUIRE(vm.env().has("ne_result1"));
    REQUIRE_THAT(vm.env().lookup("ne_result1").item<float>(), WithinAbs(1.0f, 1e-5f));
    REQUIRE(vm.env().has("ne_result2"));
    REQUIRE_THAT(vm.env().lookup("ne_result2").item<float>(), WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("Tensor comparisons - element-wise vector comparisons", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        X[0] = 1.0
        X[1] = 2.0
        X[2] = 3.0
        X[3] = 4.0
        X[4] = 5.0

        threshold = 3.0

        // Element-wise comparisons with scalar
        below[i] = X[i] < threshold
        above[i] = X[i] > threshold
        equal_to[i] = X[i] == threshold
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("below"));
    auto below = vm.env().lookup("below");
    REQUIRE_THAT(getTensorValue(below, {0}), WithinAbs(1.0f, 1e-5f));  // 1 < 3
    REQUIRE_THAT(getTensorValue(below, {1}), WithinAbs(1.0f, 1e-5f));  // 2 < 3
    REQUIRE_THAT(getTensorValue(below, {2}), WithinAbs(0.0f, 1e-5f));  // 3 < 3
    REQUIRE_THAT(getTensorValue(below, {3}), WithinAbs(0.0f, 1e-5f));  // 4 < 3
    REQUIRE_THAT(getTensorValue(below, {4}), WithinAbs(0.0f, 1e-5f));  // 5 < 3

    REQUIRE(vm.env().has("above"));
    auto above = vm.env().lookup("above");
    REQUIRE_THAT(getTensorValue(above, {0}), WithinAbs(0.0f, 1e-5f));  // 1 > 3
    REQUIRE_THAT(getTensorValue(above, {1}), WithinAbs(0.0f, 1e-5f));  // 2 > 3
    REQUIRE_THAT(getTensorValue(above, {2}), WithinAbs(0.0f, 1e-5f));  // 3 > 3
    REQUIRE_THAT(getTensorValue(above, {3}), WithinAbs(1.0f, 1e-5f));  // 4 > 3
    REQUIRE_THAT(getTensorValue(above, {4}), WithinAbs(1.0f, 1e-5f));  // 5 > 3

    REQUIRE(vm.env().has("equal_to"));
    auto equal_to = vm.env().lookup("equal_to");
    REQUIRE_THAT(getTensorValue(equal_to, {0}), WithinAbs(0.0f, 1e-5f));  // 1 == 3
    REQUIRE_THAT(getTensorValue(equal_to, {1}), WithinAbs(0.0f, 1e-5f));  // 2 == 3
    REQUIRE_THAT(getTensorValue(equal_to, {2}), WithinAbs(1.0f, 1e-5f));  // 3 == 3
    REQUIRE_THAT(getTensorValue(equal_to, {3}), WithinAbs(0.0f, 1e-5f));  // 4 == 3
    REQUIRE_THAT(getTensorValue(equal_to, {4}), WithinAbs(0.0f, 1e-5f));  // 5 == 3
}

TEST_CASE("Tensor comparisons - element-wise tensor-to-tensor", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        A[0] = 1.0
        A[1] = 3.0
        A[2] = 5.0

        B[0] = 2.0
        B[1] = 3.0
        B[2] = 4.0

        // Element-wise tensor-to-tensor comparisons
        less[i] = A[i] < B[i]
        equal[i] = A[i] == B[i]
        greater[i] = A[i] > B[i]
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("less"));
    auto less = vm.env().lookup("less");
    REQUIRE_THAT(getTensorValue(less, {0}), WithinAbs(1.0f, 1e-5f));   // 1 < 2
    REQUIRE_THAT(getTensorValue(less, {1}), WithinAbs(0.0f, 1e-5f));   // 3 < 3
    REQUIRE_THAT(getTensorValue(less, {2}), WithinAbs(0.0f, 1e-5f));   // 5 < 4

    REQUIRE(vm.env().has("equal"));
    auto equal = vm.env().lookup("equal");
    REQUIRE_THAT(getTensorValue(equal, {0}), WithinAbs(0.0f, 1e-5f));  // 1 == 2
    REQUIRE_THAT(getTensorValue(equal, {1}), WithinAbs(1.0f, 1e-5f));  // 3 == 3
    REQUIRE_THAT(getTensorValue(equal, {2}), WithinAbs(0.0f, 1e-5f));  // 5 == 4

    REQUIRE(vm.env().has("greater"));
    auto greater = vm.env().lookup("greater");
    REQUIRE_THAT(getTensorValue(greater, {0}), WithinAbs(0.0f, 1e-5f));  // 1 > 2
    REQUIRE_THAT(getTensorValue(greater, {1}), WithinAbs(0.0f, 1e-5f));  // 3 > 3
    REQUIRE_THAT(getTensorValue(greater, {2}), WithinAbs(1.0f, 1e-5f));  // 5 > 4
}

TEST_CASE("Tensor comparisons - mask creation for filtering", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        Data[0] = 10.0
        Data[1] = 25.0
        Data[2] = 5.0
        Data[3] = 30.0
        Data[4] = 15.0

        min_threshold = 10.0
        max_threshold = 25.0

        // Create boolean mask for values in range [min, max]
        in_range[i] = Data[i] >= min_threshold
        too_high[i] = Data[i] > max_threshold

        // Masked values: use multiplication with mask
        filtered[i] = Data[i] * in_range[i]
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("in_range"));
    auto in_range = vm.env().lookup("in_range");
    REQUIRE_THAT(getTensorValue(in_range, {0}), WithinAbs(1.0f, 1e-5f));  // 10 >= 10
    REQUIRE_THAT(getTensorValue(in_range, {1}), WithinAbs(1.0f, 1e-5f));  // 25 >= 10
    REQUIRE_THAT(getTensorValue(in_range, {2}), WithinAbs(0.0f, 1e-5f));  // 5 >= 10
    REQUIRE_THAT(getTensorValue(in_range, {3}), WithinAbs(1.0f, 1e-5f));  // 30 >= 10
    REQUIRE_THAT(getTensorValue(in_range, {4}), WithinAbs(1.0f, 1e-5f));  // 15 >= 10

    REQUIRE(vm.env().has("filtered"));
    auto filtered = vm.env().lookup("filtered");
    REQUIRE_THAT(getTensorValue(filtered, {0}), WithinAbs(10.0f, 1e-5f));  // 10 * 1
    REQUIRE_THAT(getTensorValue(filtered, {1}), WithinAbs(25.0f, 1e-5f));  // 25 * 1
    REQUIRE_THAT(getTensorValue(filtered, {2}), WithinAbs(0.0f, 1e-5f));   // 5 * 0
    REQUIRE_THAT(getTensorValue(filtered, {3}), WithinAbs(30.0f, 1e-5f));  // 30 * 1
    REQUIRE_THAT(getTensorValue(filtered, {4}), WithinAbs(15.0f, 1e-5f));  // 15 * 1
}

TEST_CASE("Tensor comparisons - 2D matrix comparisons", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        M[0, 0] = 1.0
        M[0, 1] = 2.0
        M[1, 0] = 3.0
        M[1, 1] = 4.0

        threshold = 2.5

        // Element-wise comparison of 2D matrix
        above_threshold[i, j] = M[i, j] > threshold
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("above_threshold"));
    auto above = vm.env().lookup("above_threshold");

    REQUIRE_THAT(getTensorValue(above, {0, 0}), WithinAbs(0.0f, 1e-5f));  // 1.0 > 2.5
    REQUIRE_THAT(getTensorValue(above, {0, 1}), WithinAbs(0.0f, 1e-5f));  // 2.0 > 2.5
    REQUIRE_THAT(getTensorValue(above, {1, 0}), WithinAbs(1.0f, 1e-5f));  // 3.0 > 2.5
    REQUIRE_THAT(getTensorValue(above, {1, 1}), WithinAbs(1.0f, 1e-5f));  // 4.0 > 2.5
}

TEST_CASE("Tensor comparisons - combining comparisons with masks", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        X[0] = 1.0
        X[1] = 5.0
        X[2] = 7.0
        X[3] = 10.0
        X[4] = 15.0

        min_val = 5.0
        max_val = 10.0

        // Create separate comparison masks
        above_min[i] = X[i] >= min_val
        below_max[i] = X[i] <= max_val

        // Combine masks via multiplication (AND logic)
        in_range[i] = above_min[i] * below_max[i]

        // For out-of-range: 1 - in_range works, but let's use explicit comparisons
        below_min[i] = X[i] < min_val
        above_max[i] = X[i] > max_val
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("in_range"));
    auto in_range = vm.env().lookup("in_range");
    REQUIRE_THAT(getTensorValue(in_range, {0}), WithinAbs(0.0f, 1e-5f));  // 1 in [5, 10]
    REQUIRE_THAT(getTensorValue(in_range, {1}), WithinAbs(1.0f, 1e-5f));  // 5 in [5, 10]
    REQUIRE_THAT(getTensorValue(in_range, {2}), WithinAbs(1.0f, 1e-5f));  // 7 in [5, 10]
    REQUIRE_THAT(getTensorValue(in_range, {3}), WithinAbs(1.0f, 1e-5f));  // 10 in [5, 10]
    REQUIRE_THAT(getTensorValue(in_range, {4}), WithinAbs(0.0f, 1e-5f));  // 15 in [5, 10]

    REQUIRE(vm.env().has("below_min"));
    auto below_min = vm.env().lookup("below_min");
    REQUIRE_THAT(getTensorValue(below_min, {0}), WithinAbs(1.0f, 1e-5f));  // 1 < 5
    REQUIRE_THAT(getTensorValue(below_min, {1}), WithinAbs(0.0f, 1e-5f));  // 5 < 5

    REQUIRE(vm.env().has("above_max"));
    auto above_max = vm.env().lookup("above_max");
    REQUIRE_THAT(getTensorValue(above_max, {4}), WithinAbs(1.0f, 1e-5f));  // 15 > 10
    REQUIRE_THAT(getTensorValue(above_max, {3}), WithinAbs(0.0f, 1e-5f));  // 10 > 10
}

TEST_CASE("Tensor comparisons - using masks in arithmetic", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        Values[0] = -5.0
        Values[1] = -2.0
        Values[2] = 0.0
        Values[3] = 3.0
        Values[4] = 8.0

        // Apply different operations based on sign
        positive_mask[i] = Values[i] > 0.0
        negative_mask[i] = Values[i] < 0.0

        // Square positive values, take absolute value of negative values
        processed[i] = (Values[i] * Values[i] * positive_mask[i]) +
                       (-1.0 * Values[i] * negative_mask[i])
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("processed"));
    auto processed = vm.env().lookup("processed");

    REQUIRE_THAT(getTensorValue(processed, {0}), WithinAbs(5.0f, 1e-5f));   // abs(-5)
    REQUIRE_THAT(getTensorValue(processed, {1}), WithinAbs(2.0f, 1e-5f));   // abs(-2)
    REQUIRE_THAT(getTensorValue(processed, {2}), WithinAbs(0.0f, 1e-5f));   // 0
    REQUIRE_THAT(getTensorValue(processed, {3}), WithinAbs(9.0f, 1e-5f));   // 3^2
    REQUIRE_THAT(getTensorValue(processed, {4}), WithinAbs(64.0f, 1e-5f));  // 8^2
}

TEST_CASE("Tensor comparisons - relu implementation using comparison", "[tensor][comparisons]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        Input[0] = -3.0
        Input[1] = -1.0
        Input[2] = 0.0
        Input[3] = 2.0
        Input[4] = 5.0

        // Manual ReLU: max(0, x) = x * (x > 0)
        manual_relu[i] = Input[i] * (Input[i] > 0.0)

        // Compare with built-in relu
        builtin_relu[i] = relu(Input[i])
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("manual_relu"));
    REQUIRE(vm.env().has("builtin_relu"));

    auto manual = vm.env().lookup("manual_relu");
    auto builtin = vm.env().lookup("builtin_relu");

    // Verify they produce the same results
    for (int i = 0; i < 5; i++) {
        float manual_val = getTensorValue(manual, {i});
        float builtin_val = getTensorValue(builtin, {i});
        REQUIRE_THAT(manual_val, WithinAbs(builtin_val, 1e-5f));
    }

    // Verify specific values
    REQUIRE_THAT(getTensorValue(manual, {0}), WithinAbs(0.0f, 1e-5f));  // max(0, -3)
    REQUIRE_THAT(getTensorValue(manual, {2}), WithinAbs(0.0f, 1e-5f));  // max(0, 0)
    REQUIRE_THAT(getTensorValue(manual, {3}), WithinAbs(2.0f, 1e-5f));  // max(0, 2)
    REQUIRE_THAT(getTensorValue(manual, {4}), WithinAbs(5.0f, 1e-5f));  // max(0, 5)
}
