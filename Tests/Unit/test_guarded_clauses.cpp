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

TEST_CASE("Guarded clauses - conditional weighting", "[guarded_clauses]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        // Input data 0..29 -> 1..30
        X[0] = 1.0
        X[1] = 2.0
        X[2] = 3.0
        X[3] = 4.0
        X[4] = 5.0
        X[5] = 6.0
        X[6] = 7.0
        X[7] = 8.0
        X[8] = 9.0
        X[9] = 10.0
        X[10] = 11.0
        X[11] = 12.0
        X[12] = 13.0
        X[13] = 14.0
        X[14] = 15.0
        X[15] = 16.0
        X[16] = 17.0
        X[17] = 18.0
        X[18] = 19.0
        X[19] = 20.0
        X[20] = 21.0
        X[21] = 22.0
        X[22] = 23.0
        X[23] = 24.0
        X[24] = 25.0
        X[25] = 26.0
        X[26] = 27.0
        X[27] = 28.0
        X[28] = 29.0
        X[29] = 30.0

        // Guarded clauses as in Examples/Programs/15_guarded_clauses.tl
        Weighted[i] = 1.0 * X[i] : (i < 10)
                    | 1.0 * X[i] : (i > 20)
                    | 0.5 * X[i] : (i >= 10 and i <= 20 and i % 2 == 0)
                    | 0.1 * X[i]
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("Weighted"));
    auto W = vm.env().lookup("Weighted");

    // Check some representative indices per example expectations
    REQUIRE_THAT(getTensorValue(W, {0}), WithinAbs(1.0f, 1e-5f));     // 1.0*1.0
    REQUIRE_THAT(getTensorValue(W, {5}), WithinAbs(6.0f, 1e-5f));     // 1.0*6.0
    REQUIRE_THAT(getTensorValue(W, {9}), WithinAbs(10.0f, 1e-5f));    // 1.0*10.0

    REQUIRE_THAT(getTensorValue(W, {10}), WithinAbs(5.5f, 1e-5f));    // 0.5*11.0
    REQUIRE_THAT(getTensorValue(W, {11}), WithinAbs(1.2f, 1e-5f));    // 0.1*12.0
    REQUIRE_THAT(getTensorValue(W, {12}), WithinAbs(6.5f, 1e-5f));    // 0.5*13.0
    REQUIRE_THAT(getTensorValue(W, {15}), WithinAbs(1.6f, 1e-5f));    // 0.1*16.0
    REQUIRE_THAT(getTensorValue(W, {20}), WithinAbs(10.5f, 1e-5f));   // 0.5*21.0

    REQUIRE_THAT(getTensorValue(W, {21}), WithinAbs(22.0f, 1e-5f));   // 1.0*22.0
    REQUIRE_THAT(getTensorValue(W, {25}), WithinAbs(26.0f, 1e-5f));   // 1.0*26.0
    REQUIRE_THAT(getTensorValue(W, {29}), WithinAbs(30.0f, 1e-5f));   // 1.0*30.0
}

TEST_CASE("Guarded clauses - piecewise function", "[guarded_clauses]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        // Input values per example 15_piecewise_function.tl
        X[0] = -5.0
        X[1] = -3.0
        X[2] = -1.0
        X[3] = 0.0
        X[4] = 1.0
        X[5] = 3.0
        X[6] = 5.0
        X[7] = 7.0

        Y[i] = X[i] * X[i] : (X[i] < 0.0)
             | 0.0 : (X[i] == 0.0)
             | sqrt(X[i]) : (X[i] > 0.0 and X[i] <= 4.0)
             | 2.0 * X[i]
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("Y"));
    auto Y = vm.env().lookup("Y");

    REQUIRE_THAT(getTensorValue(Y, {0}), WithinAbs(25.0f, 1e-4f));     // (-5)^2
    REQUIRE_THAT(getTensorValue(Y, {1}), WithinAbs(9.0f, 1e-4f));      // (-3)^2
    REQUIRE_THAT(getTensorValue(Y, {2}), WithinAbs(1.0f, 1e-4f));      // (-1)^2
    REQUIRE_THAT(getTensorValue(Y, {3}), WithinAbs(0.0f, 1e-4f));      // 0
    REQUIRE_THAT(getTensorValue(Y, {4}), WithinAbs(1.0f, 1e-4f));      // sqrt(1)
    REQUIRE_THAT(getTensorValue(Y, {5}), WithinAbs(std::sqrt(3.0f), 1e-4f)); // sqrt(3)
    REQUIRE_THAT(getTensorValue(Y, {6}), WithinAbs(10.0f, 1e-4f));     // 2*5
    REQUIRE_THAT(getTensorValue(Y, {7}), WithinAbs(14.0f, 1e-4f));     // 2*7
}

TEST_CASE("Guarded clauses - multi condition feature engineering", "[guarded_clauses]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        Age[0] = 5.0
        Age[1] = 15.0
        Age[2] = 25.0
        Age[3] = 45.0
        Age[4] = 70.0

        Income[0] = 0.0
        Income[1] = 20000.0
        Income[2] = 50000.0
        Income[3] = 80000.0
        Income[4] = 40000.0

        RiskScore[i] = 1.0 : (Age[i] < 18.0)
                     | 0.5 : (Age[i] >= 18.0 and Age[i] < 30.0 and Income[i] < 30000.0)
                     | 0.3 : (Age[i] >= 18.0 and Age[i] < 30.0 and Income[i] >= 30000.0)
                     | 0.4 : (Age[i] >= 30.0 and Age[i] < 60.0 and Income[i] < 50000.0)
                     | 0.2 : (Age[i] >= 30.0 and Age[i] < 60.0 and Income[i] >= 50000.0)
                     | 0.6 : (Age[i] >= 60.0 and Income[i] < 40000.0)
                     | 0.4 : (Age[i] >= 60.0)
        )");

    vm.execute(prog);

    REQUIRE(vm.env().has("Age"));
    REQUIRE(vm.env().has("Income"));
    REQUIRE(vm.env().has("RiskScore"));

    auto riskScore = vm.env().lookup("RiskScore");

    REQUIRE_THAT(getTensorValue(riskScore, {0}), WithinAbs(1.0f, 1e-4f));     // 1.0
    REQUIRE_THAT(getTensorValue(riskScore, {1}), WithinAbs(1.0f, 1e-4f));     // 1.0
    REQUIRE_THAT(getTensorValue(riskScore, {2}), WithinAbs(0.3f, 1e-4f));     // 0.3
    REQUIRE_THAT(getTensorValue(riskScore, {3}), WithinAbs(0.2f, 1e-4f));     // 0.2
    REQUIRE_THAT(getTensorValue(riskScore, {4}), WithinAbs(0.4f, 1e-4f));     // 0.4
}

TEST_CASE("Guarded clauses - data preprocessing guards", "[guarded_clauses]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        RawData[0] = -999.0  // Missing value sentinel
        RawData[1] = 10.0
        RawData[2] = 15.0
        RawData[3] = 200.0   // Outlier (too high)
        RawData[4] = 12.0
        RawData[5] = -999.0  // Missing value sentinel
        RawData[6] = 8.0
        RawData[7] = -50.0   // Outlier (negative)
        RawData[8] = 11.0
        RawData[9] = 14.0

        mean_value = 12.0
        min_valid = 0.0
        max_valid = 100.0

        CleanData[i] = mean_value : (RawData[i] == -999.0)
                     | min_valid : (RawData[i] < min_valid)
                     | max_valid : (RawData[i] > max_valid)
                     | RawData[i]
        )");

    vm.execute(prog);

    REQUIRE(vm.env().has("RawData"));
    REQUIRE(vm.env().has("mean_value"));
    REQUIRE(vm.env().has("min_valid"));
    REQUIRE(vm.env().has("max_valid"));
    REQUIRE(vm.env().has("CleanData"));

    auto cleanData = vm.env().lookup("CleanData");

    REQUIRE_THAT(getTensorValue(cleanData, {0}), WithinAbs(12.0f, 1e-4f));     // 12.0
    REQUIRE_THAT(getTensorValue(cleanData, {1}), WithinAbs(10.0f, 1e-4f));     // 10.0
    REQUIRE_THAT(getTensorValue(cleanData, {2}), WithinAbs(15.0f, 1e-4f));     // 15.0
    REQUIRE_THAT(getTensorValue(cleanData, {3}), WithinAbs(100.0f, 1e-4f));    // 100.0
    REQUIRE_THAT(getTensorValue(cleanData, {4}), WithinAbs(12.0f, 1e-4f));     // 12.0
    REQUIRE_THAT(getTensorValue(cleanData, {5}), WithinAbs(12.0f, 1e-4f));     // 12.0
    REQUIRE_THAT(getTensorValue(cleanData, {6}), WithinAbs(8.0f, 1e-4f));      // 8.0
    REQUIRE_THAT(getTensorValue(cleanData, {7}), WithinAbs(0.0f, 1e-4f));      // 0.0
    REQUIRE_THAT(getTensorValue(cleanData, {8}), WithinAbs(11.0f, 1e-4f));     // 11.0
    REQUIRE_THAT(getTensorValue(cleanData, {9}), WithinAbs(14.0f, 1e-4f));     // 14.0
}

TEST_CASE("Guarded clauses - conditional activation function", "[guarded_clauses]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        Activations[0] = -10.0
        Activations[1] = -2.0
        Activations[2] = -0.5
        Activations[3] = 0.0
        Activations[4] = 0.5
        Activations[5] = 2.0
        Activations[6] = 10.0

        Output[i] = 0.0 : (Activations[i] < -5.0)
                  | Activations[i] + 5.0 : (Activations[i] >= -5.0 and Activations[i] < -1.0)
                  | Activations[i] : (Activations[i] >= -1.0 and Activations[i] <= 1.0)
                  | relu(Activations[i]) : (Activations[i] > 1.0 and Activations[i] < 5.0)
                  | 5.0
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("Activations"));
    REQUIRE(vm.env().has("Output"));

    auto output = vm.env().lookup("Output");

    REQUIRE_THAT(getTensorValue(output, {0}), WithinAbs(0.0f, 1e-4f));     // 0.0
    REQUIRE_THAT(getTensorValue(output, {1}), WithinAbs(3.0f, 1e-4f));     // 3.0
    REQUIRE_THAT(getTensorValue(output, {2}), WithinAbs(-0.5f, 1e-4f));    // -0.5
    REQUIRE_THAT(getTensorValue(output, {3}), WithinAbs(0.0f, 1e-4f));     // 0.0
    REQUIRE_THAT(getTensorValue(output, {4}), WithinAbs(0.5f, 1e-4f));     // 0.5
    REQUIRE_THAT(getTensorValue(output, {5}), WithinAbs(2.0f, 1e-4f));     // 2.0
    REQUIRE_THAT(getTensorValue(output, {6}), WithinAbs(5.0f, 1e-4f));     // 5.0
}

TEST_CASE("Guarded clauses - time series filtering", "[guarded_clauses]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};

    auto prog = parseProgram(R"(
        Hour[0] = 2.0   // Night
        Hour[1] = 8.0   // Morning rush
        Hour[2] = 14.0  // Afternoon
        Hour[3] = 18.0  // Evening rush
        Hour[4] = 22.0  // Night

        Readings[0] = 10.0
        Readings[1] = 50.0
        Readings[2] = 30.0
        Readings[3] = 60.0
        Readings[4] = 15.0

        Smoothed[i] = 0.9 * Readings[i] : (Hour[i] >= 0.0 and Hour[i] < 6.0)
                    | 0.9 * Readings[i] : (Hour[i] >= 22.0)
                    | 0.5 * Readings[i] : (Hour[i] >= 7.0 and Hour[i] <= 9.0)
                    | 0.5 * Readings[i] : (Hour[i] >= 17.0 and Hour[i] <= 19.0)
                    | 0.7 * Readings[i]
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("Hour"));
    REQUIRE(vm.env().has("Readings"));
    REQUIRE(vm.env().has("Smoothed"));

    auto smoothed = vm.env().lookup("Smoothed");

    REQUIRE_THAT(getTensorValue(smoothed, {0}), WithinAbs(9.0f, 1e-4f));      // 9.0
    REQUIRE_THAT(getTensorValue(smoothed, {1}), WithinAbs(25.0f, 1e-4f));     // 25.0
    REQUIRE_THAT(getTensorValue(smoothed, {2}), WithinAbs(21.0f, 1e-4f));     // 21.0
    REQUIRE_THAT(getTensorValue(smoothed, {3}), WithinAbs(30.0f, 1e-4f));     // 30.0
    REQUIRE_THAT(getTensorValue(smoothed, {4}), WithinAbs(13.5f, 1e-4f));     // 13.5
}
