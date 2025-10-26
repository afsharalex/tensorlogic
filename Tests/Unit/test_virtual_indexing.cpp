#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <cmath>
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


TEST_CASE("Virtual indexing - RNN", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        W[0, 0] = 0.5
        W[0, 1] = 0.2
        W[0, 2] = 0.1
        W[0, 3] = 0.3
        W[1, 0] = 0.3
        W[1, 1] = 0.6
        W[1, 2] = 0.2
        W[1, 3] = 0.1
        W[2, 0] = 0.2
        W[2, 1] = 0.1
        W[2, 2] = 0.5
        W[2, 3] = 0.4
        W[3, 0] = 0.4
        W[3, 1] = 0.3
        W[3, 2] = 0.2
        W[3, 3] = 0.6

        U[0, 0] = 0.7
        U[0, 1] = 0.3
        U[0, 2] = 0.2
        U[0, 3] = 0.5
        U[1, 0] = 0.4
        U[1, 1] = 0.6
        U[1, 2] = 0.1
        U[1, 3] = 0.3
        U[2, 0] = 0.5
        U[2, 1] = 0.2
        U[2, 2] = 0.8
        U[2, 3] = 0.4
        U[3, 0] = 0.3
        U[3, 1] = 0.5
        U[3, 2] = 0.4
        U[3, 3] = 0.6

        // Bias
        b[0] = 0.1
        b[1] = 0.2
        b[2] = 0.1
        b[3] = 0.3

        // Time step 0
        Input[0, 0] = 1.0
        Input[1, 0] = 0.5
        Input[2, 0] = 0.8
        Input[3, 0] = 0.3

        // Time step 1
        Input[0, 1] = 0.8
        Input[1, 1] = 0.6
        Input[2, 1] = 0.7
        Input[3, 1] = 0.4

        // Time step 2
        Input[0, 2] = 0.6
        Input[1, 2] = 0.9
        Input[2, 2] = 0.5
        Input[3, 2] = 0.7

        // Time step 3
        Input[0, 3] = 0.9
        Input[1, 3] = 0.4
        Input[2, 3] = 0.8
        Input[3, 3] = 0.5

        // Time step 4
        Input[0, 4] = 0.7
        Input[1, 4] = 0.7
        Input[2, 4] = 0.6
        Input[3, 4] = 0.8

        // State at t=0
        State[0, 0] = 0.0
        State[1, 0] = 0.0
        State[2, 0] = 0.0
        State[3, 0] = 0.0

        State[i, *t+1] = relu(
            W[i, j] State[j, *t]
          + U[i, j] Input[j, t]
          + b[i]
        )

        W_out[0] = 0.6
        W_out[1] = 0.4
        W_out[2] = 0.5
        W_out[3] = 0.3

        Output = sigmoid(W_out[i] State[i, 0])

        State[0, 0]?
        State[1, 0]?
        State[2, 0]?
        State[3, 0]?
)");
    vm.execute(prog);

    REQUIRE(vm.env().has("Output"));
    REQUIRE(vm.env().has("State"));

    auto state = vm.env().lookup("State");

    REQUIRE_THAT(getTensorValue(state, {0, 0}), WithinAbs(9.35f, 0.001f));
    REQUIRE_THAT(getTensorValue(state, {1, 0}), WithinAbs(9.501f, 0.001f));
    REQUIRE_THAT(getTensorValue(state, {2, 0}), WithinAbs(10.713f, 0.001f));
    REQUIRE_THAT(getTensorValue(state, {3, 0}), WithinAbs(13.050f, 0.001f));
}

TEST_CASE("Virtual indexing - RNN projection layer", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        W[0, 0] = 0.5
        W[0, 1] = 0.2
        W[0, 2] = 0.1
        W[0, 3] = 0.3
        W[1, 0] = 0.3
        W[1, 1] = 0.6
        W[1, 2] = 0.2
        W[1, 3] = 0.1
        W[2, 0] = 0.2
        W[2, 1] = 0.1
        W[2, 2] = 0.5
        W[2, 3] = 0.4
        W[3, 0] = 0.4
        W[3, 1] = 0.3
        W[3, 2] = 0.2
        W[3, 3] = 0.6

        U_proj[0, 0] = 0.7
        U_proj[0, 1] = 0.3
        U_proj[0, 2] = 0.2
        U_proj[1, 0] = 0.4
        U_proj[1, 1] = 0.6
        U_proj[1, 2] = 0.1
        U_proj[2, 0] = 0.5
        U_proj[2, 1] = 0.2
        U_proj[2, 2] = 0.8
        U_proj[3, 0] = 0.3
        U_proj[3, 1] = 0.5
        U_proj[3, 2] = 0.4

        b[0] = 0.1
        b[1] = 0.2
        b[2] = 0.1
        b[3] = 0.3

        // Time step 0
        Input[0, 0] = 1.0
        Input[1, 0] = 0.5
        Input[2, 0] = 0.8

        // Time step 1
        Input[0, 1] = 0.8
        Input[1, 1] = 0.6
        Input[2, 1] = 0.7

        // Time step 2
        Input[0, 2] = 0.6
        Input[1, 2] = 0.9
        Input[2, 2] = 0.5

        // Time step 3
        Input[0, 3] = 0.9
        Input[1, 3] = 0.4
        Input[2, 3] = 0.8

        // Time step 4
        Input[0, 4] = 0.7
        Input[1, 4] = 0.7
        Input[2, 4] = 0.6

        State[0, 0] = 0.0
        State[1, 0] = 0.0
        State[2, 0] = 0.0
        State[3, 0] = 0.0

        Input_proj[i, t] = U_proj[i, k] Input[k, t]

        State[i, *t+1] = relu(
            W[i, j] State[j, *t]
          + Input_proj[i, t]
          + b[i]
        )

        W_out[0] = 0.6
        W_out[1] = 0.4
        W_out[2] = 0.5
        W_out[3] = 0.3

        bias_out = -0.2

        Output = sigmoid(W_out[i] State[i, *5] + bias_out)

        Input_proj[0, 0]?  // First hidden unit, first time step
        Input_proj[1, 2]?  // Second hidden unit, third time step
        Input_proj[i, 0]?  // All hidden units at first time step

        State[0, *5]?  // First hidden unit's final value
        State[1, *5]?  // Second hidden unit's final value
        State[2, *5]?  // Third hidden unit's final value
        State[3, *5]?  // Fourth hidden unit's final value
        State[i, *5]?  // All hidden units' final values
)");
    vm.execute(prog);

    REQUIRE(vm.env().has("Output"));
    REQUIRE(vm.env().has("State"));

    auto state = vm.env().lookup("State");

    REQUIRE_THAT(getTensorValue(state, {0, 0}), WithinAbs(7.630f, 0.001f));
    REQUIRE_THAT(getTensorValue(state, {1, 0}), WithinAbs(7.960f, 0.001f));
    REQUIRE_THAT(getTensorValue(state, {2, 0}), WithinAbs(8.886f, 0.001f));
    REQUIRE_THAT(getTensorValue(state, {3, 0}), WithinAbs(10.694f, 0.001f));
}

TEST_CASE("Virtual indexing - RNN multi layer projection", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        U1[0, 0] = 0.7
        U1[0, 1] = 0.3
        U1[0, 2] = 0.2
        U1[1, 0] = 0.4
        U1[1, 1] = 0.6
        U1[1, 2] = 0.1
        U1[2, 0] = 0.5
        U1[2, 1] = 0.2
        U1[2, 2] = 0.8
        U1[3, 0] = 0.3
        U1[3, 1] = 0.5
        U1[3, 2] = 0.4

        // Recurrent weights layer 1: 4 × 4
        W1[0, 0] = 0.5
        W1[0, 1] = 0.2
        W1[0, 2] = 0.1
        W1[0, 3] = 0.3
        W1[1, 0] = 0.3
        W1[1, 1] = 0.6
        W1[1, 2] = 0.2
        W1[1, 3] = 0.1
        W1[2, 0] = 0.2
        W1[2, 1] = 0.1
        W1[2, 2] = 0.5
        W1[2, 3] = 0.4
        W1[3, 0] = 0.4
        W1[3, 1] = 0.3
        W1[3, 2] = 0.2
        W1[3, 3] = 0.6

        b1[0] = 0.1
        b1[1] = 0.2
        b1[2] = 0.1
        b1[3] = 0.3

        U2[0, 0] = 0.6
        U2[0, 1] = 0.3
        U2[0, 2] = 0.4
        U2[0, 3] = 0.2
        U2[1, 0] = 0.5
        U2[1, 1] = 0.4
        U2[1, 2] = 0.3
        U2[1, 3] = 0.5
        U2[2, 0] = 0.4
        U2[2, 1] = 0.5
        U2[2, 2] = 0.6
        U2[2, 3] = 0.3

        // Recurrent weights layer 2: 3 × 3
        W2[0, 0] = 0.6
        W2[0, 1] = 0.2
        W2[0, 2] = 0.3
        W2[1, 0] = 0.3
        W2[1, 1] = 0.7
        W2[1, 2] = 0.2
        W2[2, 0] = 0.4
        W2[2, 1] = 0.3
        W2[2, 2] = 0.6

        b2[0] = 0.2
        b2[1] = 0.1
        b2[2] = 0.3

        Input[0, 0] = 1.0
        Input[1, 0] = 0.5
        Input[2, 0] = 0.8

        Input[0, 1] = 0.8
        Input[1, 1] = 0.6
        Input[2, 1] = 0.7

        Input[0, 2] = 0.6
        Input[1, 2] = 0.9
        Input[2, 2] = 0.5

        Input[0, 3] = 0.9
        Input[1, 3] = 0.4
        Input[2, 3] = 0.8

        State1[0, 0] = 0.0
        State1[1, 0] = 0.0
        State1[2, 0] = 0.0
        State1[3, 0] = 0.0

        State2[0, 0] = 0.0
        State2[1, 0] = 0.0
        State2[2, 0] = 0.0

        Input_proj1[i, t] = U1[i, k] Input[k, t]

        // RNN layer 1
        State1[i, *t+1] = relu(
            W1[i, j] State1[j, *t]
          + Input_proj1[i, t]
          + b1[i]
        )

        Input_proj2[i, t] = U2[i, j] State1[j, *t+1]

        // RNN layer 2
        State2[i, *t+1] = relu(
            W2[i, j] State2[j, *t]
          + Input_proj2[i, t]
          + b2[i]
        )

        W_out[0] = 0.5
        W_out[1] = 0.4
        W_out[2] = 0.6

        Output = sigmoid(W_out[i] State2[i, *4])
    )");
    vm.execute(prog);

    REQUIRE(vm.env().has("Output"));
    REQUIRE(vm.env().has("State1"));
    REQUIRE(vm.env().has("State2"));

    auto output = vm.env().lookup("Output");
    auto state1 = vm.env().lookup("State1");
    auto state2 = vm.env().lookup("State2");

    // Final output should be ~0.5 after sigmoid activation
    REQUIRE_THAT(output.item<float>(), WithinAbs(0.5f, 0.001f));

    // Test final states of both RNN layers are non-negative due to ReLU
    REQUIRE(getTensorValue(state1, {0, 0}) >= 0.0f);
    REQUIRE(getTensorValue(state1, {1, 0}) >= 0.0f);
    REQUIRE(getTensorValue(state2, {0, 0}) >= 0.0f);
    REQUIRE(getTensorValue(state2, {1, 0}) >= 0.0f);
}

TEST_CASE("Virtual indexing - RNN bidirectional input projection", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        U_fwd[0, 0] = 0.7
        U_fwd[0, 1] = 0.3
        U_fwd[0, 2] = 0.2
        U_fwd[1, 0] = 0.4
        U_fwd[1, 1] = 0.6
        U_fwd[1, 2] = 0.1
        U_fwd[2, 0] = 0.5
        U_fwd[2, 1] = 0.2
        U_fwd[2, 2] = 0.8
        U_fwd[3, 0] = 0.3
        U_fwd[3, 1] = 0.5
        U_fwd[3, 2] = 0.4

        W_fwd[0, 0] = 0.5
        W_fwd[0, 1] = 0.2
        W_fwd[0, 2] = 0.1
        W_fwd[0, 3] = 0.3
        W_fwd[1, 0] = 0.3
        W_fwd[1, 1] = 0.6
        W_fwd[1, 2] = 0.2
        W_fwd[1, 3] = 0.1
        W_fwd[2, 0] = 0.2
        W_fwd[2, 1] = 0.1
        W_fwd[2, 2] = 0.5
        W_fwd[2, 3] = 0.4
        W_fwd[3, 0] = 0.4
        W_fwd[3, 1] = 0.3
        W_fwd[3, 2] = 0.2
        W_fwd[3, 3] = 0.6

        U_bwd[0, 0] = 0.6
        U_bwd[0, 1] = 0.4
        U_bwd[0, 2] = 0.3
        U_bwd[1, 0] = 0.5
        U_bwd[1, 1] = 0.5
        U_bwd[1, 2] = 0.2
        U_bwd[2, 0] = 0.4
        U_bwd[2, 1] = 0.3
        U_bwd[2, 2] = 0.7
        U_bwd[3, 0] = 0.4
        U_bwd[3, 1] = 0.6
        U_bwd[3, 2] = 0.3

        W_bwd[0, 0] = 0.6
        W_bwd[0, 1] = 0.3
        W_bwd[0, 2] = 0.2
        W_bwd[0, 3] = 0.4
        W_bwd[1, 0] = 0.4
        W_bwd[1, 1] = 0.5
        W_bwd[1, 2] = 0.3
        W_bwd[1, 3] = 0.2
        W_bwd[2, 0] = 0.3
        W_bwd[2, 1] = 0.2
        W_bwd[2, 2] = 0.6
        W_bwd[2, 3] = 0.3
        W_bwd[3, 0] = 0.5
        W_bwd[3, 1] = 0.4
        W_bwd[3, 2] = 0.3
        W_bwd[3, 3] = 0.5

        Input[0, 0] = 1.0
        Input[1, 0] = 0.5
        Input[2, 0] = 0.8

        Input[0, 1] = 0.8
        Input[1, 1] = 0.6
        Input[2, 1] = 0.7

        Input[0, 2] = 0.6
        Input[1, 2] = 0.9
        Input[2, 2] = 0.5

        // Create reversed input for backward pass
        Input_rev[k, 0] = Input[k, 2]
        Input_rev[k, 1] = Input[k, 1]
        Input_rev[k, 2] = Input[k, 0]

        State_fwd[0, 0] = 0.0
        State_fwd[1, 0] = 0.0
        State_fwd[2, 0] = 0.0
        State_fwd[3, 0] = 0.0

        State_bwd[0, 0] = 0.0
        State_bwd[1, 0] = 0.0
        State_bwd[2, 0] = 0.0
        State_bwd[3, 0] = 0.0

        Input_proj_fwd[i, t] = U_fwd[i, k] Input[k, t]

        State_fwd[i, *t+1] = tanh(
            W_fwd[i, j] State_fwd[j, *t]
          + Input_proj_fwd[i, t]
        )

        Input_proj_bwd[i, t] = U_bwd[i, k] Input_rev[k, t]

        State_bwd[i, *t+1] = tanh(
            W_bwd[i, j] State_bwd[j, *t]
          + Input_proj_bwd[i, t]
        )

        Combined[0] = State_fwd[0, *3]
        Combined[1] = State_fwd[1, *3]
        Combined[2] = State_fwd[2, *3]
        Combined[3] = State_fwd[3, *3]
        Combined[4] = State_bwd[0, *3]
        Combined[5] = State_bwd[1, *3]
        Combined[6] = State_bwd[2, *3]
        Combined[7] = State_bwd[3, *3]

        // Output from combined representation
        W_out[0] = 0.3
        W_out[1] = 0.4
        W_out[2] = 0.2
        W_out[3] = 0.3
        W_out[4] = 0.4
        W_out[5] = 0.3
        W_out[6] = 0.5
        W_out[7] = 0.2

        Output = sigmoid(W_out[i] Combined[i])
    )");
    vm.execute(prog);

    REQUIRE(vm.env().has("Output"));
    REQUIRE(vm.env().has("Combined"));

    auto output = vm.env().lookup("Output");
    auto combined = vm.env().lookup("Combined");

    // Final output should be ~0.5 after sigmoid activation
    REQUIRE_THAT(output.item<float>(), WithinAbs(0.5f, 0.001f));

    // Since bidirectional RNN uses tanh, combined state values should be in [-1, 1]
    REQUIRE(combined.item<float>() >= -1.0f);
    REQUIRE(combined.item<float>() <= 1.0f);
}

TEST_CASE("Virtual indexing - LSTM cell", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        W_f[0, 0] = 0.5
        W_f[0, 1] = 0.3
        W_f[0, 2] = 0.2
        W_f[1, 0] = 0.4
        W_f[1, 1] = 0.6
        W_f[1, 2] = 0.1
        W_f[2, 0] = 0.3
        W_f[2, 1] = 0.2
        W_f[2, 2] = 0.5

        U_f[0, 0] = 0.6
        U_f[0, 1] = 0.4
        U_f[1, 0] = 0.5
        U_f[1, 1] = 0.3
        U_f[2, 0] = 0.4
        U_f[2, 1] = 0.6

        // Input gate weights
        W_i[0, 0] = 0.4
        W_i[0, 1] = 0.5
        W_i[0, 2] = 0.3
        W_i[1, 0] = 0.6
        W_i[1, 1] = 0.2
        W_i[1, 2] = 0.4
        W_i[2, 0] = 0.3
        W_i[2, 1] = 0.5
        W_i[2, 2] = 0.2

        U_i[0, 0] = 0.5
        U_i[0, 1] = 0.5
        U_i[1, 0] = 0.4
        U_i[1, 1] = 0.6
        U_i[2, 0] = 0.6
        U_i[2, 1] = 0.3

        // Cell candidate weights
        W_c[0, 0] = 0.6
        W_c[0, 1] = 0.2
        W_c[0, 2] = 0.4
        W_c[1, 0] = 0.3
        W_c[1, 1] = 0.7
        W_c[1, 2] = 0.2
        W_c[2, 0] = 0.5
        W_c[2, 1] = 0.3
        W_c[2, 2] = 0.6

        U_c[0, 0] = 0.7
        U_c[0, 1] = 0.3
        U_c[1, 0] = 0.4
        U_c[1, 1] = 0.8
        U_c[2, 0] = 0.6
        U_c[2, 1] = 0.5

        // Output gate weights
        W_o[0, 0] = 0.5
        W_o[0, 1] = 0.4
        W_o[0, 2] = 0.3
        W_o[1, 0] = 0.6
        W_o[1, 1] = 0.3
        W_o[1, 2] = 0.5
        W_o[2, 0] = 0.4
        W_o[2, 1] = 0.6
        W_o[2, 2] = 0.4

        U_o[0, 0] = 0.6
        U_o[0, 1] = 0.5
        U_o[1, 0] = 0.5
        U_o[1, 1] = 0.4
        U_o[2, 0] = 0.7
        U_o[2, 1] = 0.4

        Input[0, 0] = 1.0
        Input[1, 0] = 0.5

        Input[0, 1] = 0.8
        Input[1, 1] = 0.7

        Input[0, 2] = 0.6
        Input[1, 2] = 0.9

        Input[0, 3] = 0.9
        Input[1, 3] = 0.6

        HiddenState[0, 0] = 0.0
        HiddenState[1, 0] = 0.0
        HiddenState[2, 0] = 0.0

        CellState[0, 0] = 0.0
        CellState[1, 0] = 0.0
        CellState[2, 0] = 0.0

        // Forget gate: decides what to forget from cell state
        ForgetGate[i, *t+1] = sigmoid(
            U_f[i, k] Input[k, t]
        )

        // Input gate: decides what new information to store
        InputGate[i, *t+1] = sigmoid(
            U_i[i, k] Input[k, t]
        )

        // Cell candidate: new candidate values
        CellCandidate[i, *t+1] = tanh(
            U_c[i, k] Input[k, t]
        )

        // Update cell state: forget old + remember new
        CellState[i, *t+1] = ForgetGate[i, *t+1] * CellState[i, *t]
                           + InputGate[i, *t+1] * CellCandidate[i, *t+1]

        // Output gate: decides what to output
        OutputGate[i, *t+1] = sigmoid(
            U_o[i, k] Input[k, t]
        )

        // Update hidden state
        HiddenState[i, *t+1] = OutputGate[i, *t+1] * tanh(CellState[i, *t+1])

)");
    
    vm.execute(prog);

    REQUIRE(vm.env().has("HiddenState"));
    REQUIRE(vm.env().has("ForgetGate"));
    REQUIRE(vm.env().has("InputGate"));
    REQUIRE(vm.env().has("OutputGate"));
    REQUIRE(vm.env().has("CellCandidate"));

    auto hiddenState = vm.env().lookup("HiddenState");
    auto forgetGate = vm.env().lookup("ForgetGate");
    auto inputGate = vm.env().lookup("InputGate");
    auto outputGate = vm.env().lookup("OutputGate");
    auto cellCandidate = vm.env().lookup("CellCandidate");

    // HiddenState is outputGate * tanh(CellState), so it must be in [-1, 1]
    REQUIRE(getTensorValue(hiddenState, {0, 0}) >= -1.0f);
    REQUIRE(getTensorValue(hiddenState, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(hiddenState, {1, 0}) >= -1.0f);
    REQUIRE(getTensorValue(hiddenState, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(hiddenState, {2, 0}) >= -1.0f);
    REQUIRE(getTensorValue(hiddenState, {2, 0}) <= 1.0f);

    // Sigmoid gates must be in [0, 1]
    REQUIRE(getTensorValue(forgetGate, {0, 0}) >= 0.0f);
    REQUIRE(getTensorValue(forgetGate, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(forgetGate, {1, 0}) >= 0.0f);
    REQUIRE(getTensorValue(forgetGate, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(forgetGate, {2, 0}) >= 0.0f);
    REQUIRE(getTensorValue(forgetGate, {2, 0}) <= 1.0f);

    REQUIRE(getTensorValue(inputGate, {0, 0}) >= 0.0f);
    REQUIRE(getTensorValue(inputGate, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(inputGate, {1, 0}) >= 0.0f);
    REQUIRE(getTensorValue(inputGate, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(inputGate, {2, 0}) >= 0.0f);
    REQUIRE(getTensorValue(inputGate, {2, 0}) <= 1.0f);

    REQUIRE(getTensorValue(outputGate, {0, 0}) >= 0.0f);
    REQUIRE(getTensorValue(outputGate, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(outputGate, {1, 0}) >= 0.0f);
    REQUIRE(getTensorValue(outputGate, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(outputGate, {2, 0}) >= 0.0f);
    REQUIRE(getTensorValue(outputGate, {2, 0}) <= 1.0f);

    // Cell candidate is tanh(...), so it must be in [-1, 1]
    REQUIRE(getTensorValue(cellCandidate, {0, 0}) >= -1.0f);
    REQUIRE(getTensorValue(cellCandidate, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(cellCandidate, {1, 0}) >= -1.0f);
    REQUIRE(getTensorValue(cellCandidate, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(cellCandidate, {2, 0}) >= -1.0f);
    REQUIRE(getTensorValue(cellCandidate, {2, 0}) <= 1.0f);
}

TEST_CASE("Virtual indexing - GRU", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Reset gate weights
        W_r[0, 0] = 0.5
        W_r[0, 1] = 0.3
        W_r[0, 2] = 0.2
        W_r[1, 0] = 0.4
        W_r[1, 1] = 0.6
        W_r[1, 2] = 0.3
        W_r[2, 0] = 0.3
        W_r[2, 1] = 0.4
        W_r[2, 2] = 0.5

        U_r[0, 0] = 0.6
        U_r[0, 1] = 0.4
        U_r[1, 0] = 0.5
        U_r[1, 1] = 0.5
        U_r[2, 0] = 0.4
        U_r[2, 1] = 0.6

        // Update gate weights
        W_z[0, 0] = 0.4
        W_z[0, 1] = 0.5
        W_z[0, 2] = 0.3
        W_z[1, 0] = 0.6
        W_z[1, 1] = 0.3
        W_z[1, 2] = 0.4
        W_z[2, 0] = 0.5
        W_z[2, 1] = 0.4
        W_z[2, 2] = 0.3

        U_z[0, 0] = 0.5
        U_z[0, 1] = 0.5
        U_z[1, 0] = 0.6
        U_z[1, 1] = 0.4
        U_z[2, 0] = 0.4
        U_z[2, 1] = 0.6

        // Candidate weights
        W_h[0, 0] = 0.6
        W_h[0, 1] = 0.3
        W_h[0, 2] = 0.4
        W_h[1, 0] = 0.4
        W_h[1, 1] = 0.7
        W_h[1, 2] = 0.2
        W_h[2, 0] = 0.5
        W_h[2, 1] = 0.4
        W_h[2, 2] = 0.6

        U_h[0, 0] = 0.7
        U_h[0, 1] = 0.3
        U_h[1, 0] = 0.5
        U_h[1, 1] = 0.6
        U_h[2, 0] = 0.6
        U_h[2, 1] = 0.5

        // Input sequence
        Input[0, 0] = 1.0
        Input[1, 0] = 0.5
        Input[0, 1] = 0.8
        Input[1, 1] = 0.6
        Input[0, 2] = 0.7
        Input[1, 2] = 0.9

        // Initial state
        h[0, 0] = 0.0
        h[1, 0] = 0.0
        h[2, 0] = 0.0

        // Reset gate: determines how much past to forget
        r[i, *t+1] = sigmoid(
            W_r[i, j] h[j, *t]
          + U_r[i, k] Input[k, t]
        )

        // Update gate: determines how much to update
        z[i, *t+1] = sigmoid(
            W_z[i, j] h[j, *t]
          + U_z[i, k] Input[k, t]
        )

        // Candidate hidden state (with reset applied)
        h_candidate[i, *t+1] = tanh(
            W_h[i, j] (r[j, *t+1] * h[j, *t])
          + U_h[i, k] Input[k, t]
        )

        // Final hidden state (interpolate between old and candidate)
        h[i, *t+1] = (1.0 - z[i, *t+1]) * h[i, *t]
                   + z[i, *t+1] * h_candidate[i, *t+1]
    )");
    
    vm.execute(prog);

    REQUIRE(vm.env().has("h"));
    REQUIRE(vm.env().has("r"));
    REQUIRE(vm.env().has("z"));
    REQUIRE(vm.env().has("h_candidate"));

    auto h = vm.env().lookup("h");
    auto r = vm.env().lookup("r");
    auto z = vm.env().lookup("z");
    auto h_candidate = vm.env().lookup("h_candidate");

    // GRU hidden state h is tanh-based and gated, so it must be in [-1, 1]
    REQUIRE(getTensorValue(h, {0, 0}) >= -1.0f);
    REQUIRE(getTensorValue(h, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(h, {1, 0}) >= -1.0f);
    REQUIRE(getTensorValue(h, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(h, {2, 0}) >= -1.0f);
    REQUIRE(getTensorValue(h, {2, 0}) <= 1.0f);

    // Sigmoid gates r and z must be in [0, 1]
    REQUIRE(getTensorValue(r, {0, 0}) >= 0.0f);
    REQUIRE(getTensorValue(r, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(r, {1, 0}) >= 0.0f);
    REQUIRE(getTensorValue(r, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(r, {2, 0}) >= 0.0f);
    REQUIRE(getTensorValue(r, {2, 0}) <= 1.0f);

    REQUIRE(getTensorValue(z, {0, 0}) >= 0.0f);
    REQUIRE(getTensorValue(z, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(z, {1, 0}) >= 0.0f);
    REQUIRE(getTensorValue(z, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(z, {2, 0}) >= 0.0f);
    REQUIRE(getTensorValue(z, {2, 0}) <= 1.0f);

    // Candidate hidden state is tanh(...), so it must be in [-1, 1]
    REQUIRE(getTensorValue(h_candidate, {0, 0}) >= -1.0f);
    REQUIRE(getTensorValue(h_candidate, {0, 0}) <= 1.0f);
    REQUIRE(getTensorValue(h_candidate, {1, 0}) >= -1.0f);
    REQUIRE(getTensorValue(h_candidate, {1, 0}) <= 1.0f);
    REQUIRE(getTensorValue(h_candidate, {2, 0}) >= -1.0f);
    REQUIRE(getTensorValue(h_candidate, {2, 0}) <= 1.0f);
}


TEST_CASE("Virtual indexing - Iterative algorithm", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        x[0] = 1.0  // Starting point

        // Fixed-point iteration: x^{t+1} = cos(x^t)
        x[*t+1] = cos(x[*t])

        // Materialize value after convergence
        x[*0]?
    )");

    vm.execute(prog);

    REQUIRE(vm.env().has("x"));

    auto x = vm.env().lookup("x");

    // Fixed-point of x = cos(x) is approximately 0.739085
    // Allow a slightly wider tolerance because the number of iterations is implementation-defined
    REQUIRE_THAT(x.item<float>(), WithinAbs(0.739f, 0.01f));
}

TEST_CASE("Virtual indexing - Exponential moving average", "[virtual_indexing]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Learning rate
        alpha = 0.1

        // Initial average
        avg[0] = 0.0

        // Streaming data points
        data[0] = 5.0
        data[1] = 8.0
        data[2] = 6.0
        data[3] = 9.0
        data[4] = 7.0

        // Exponential moving average update
        // avg^{t+1} = (1 - α) * avg^t + α * data[t]
        avg[*t+1] = (1.0 - alpha) * avg[*t] + alpha * data[t]

        // This computes:
        // t=0: avg[*1] = 0.9*0.0 + 0.1*5.0 = 0.5
        // t=1: avg[*2] = 0.9*0.5 + 0.1*8.0 = 1.25
        // t=2: avg[*3] = 0.9*1.25 + 0.1*6.0 = 1.725
        // t=3: avg[*4] = 0.9*1.725 + 0.1*9.0 = 2.4525
        // t=4: avg[*5] = 0.9*2.4525 + 0.1*7.0 = 2.90725

        // Materialize final value
        avg[*0]?
    )" );

    vm.execute(prog);

    REQUIRE(vm.env().has("alpha"));
    REQUIRE(vm.env().has("data"));
    REQUIRE(vm.env().has("avg"));

    auto avg = vm.env().lookup("avg");

    REQUIRE_THAT(avg.item<float>(), WithinAbs(2.907f, 0.001f));
}
