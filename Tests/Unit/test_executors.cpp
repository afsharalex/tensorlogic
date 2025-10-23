#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "TL/vm.hpp"
#include "TL/Parser.hpp"
#include "TL/Runtime/Executors/ScalarAssignExecutor.hpp"
#include "TL/Runtime/Executors/ListLiteralExecutor.hpp"
#include "TL/Runtime/Executors/EinsumExecutor.hpp"
#include "TL/Runtime/Executors/IndexedProductExecutor.hpp"
#include "TL/Runtime/Executors/ReductionExecutor.hpp"
#include "TL/Runtime/Executors/PoolingExecutor.hpp"
#include "TL/Runtime/Executors/IdentityExecutor.hpp"
#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include "TL/backend.hpp"

using namespace tl;

// Helper to create a backend instance for tests
static std::unique_ptr<TensorBackend> createBackend() {
    return BackendFactory::create(BackendType::LibTorch);
}

// Helper to parse a single tensor equation
static TensorEquation parseEquation(const std::string& code) {
    auto program = parseProgram(code);
    REQUIRE(program.statements.size() == 1);
    auto* eq = std::get_if<TensorEquation>(&program.statements[0]);
    REQUIRE(eq != nullptr);
    return *eq;
}

// ============================================================================
// ScalarAssignExecutor Tests
// ============================================================================

TEST_CASE("ScalarAssignExecutor::canExecute", "[executor][scalar]") {
    ScalarAssignExecutor executor;
    Environment env;

    SECTION("Recognizes indexed LHS with numeric literal RHS") {
        auto eq = parseEquation("W[0, 1] = 2.0");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes mixed numeric and zero indices") {
        auto eq = parseEquation("W[1, 0] = 1.5");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Rejects unindexed LHS") {
        auto eq = parseEquation("W = 2.0");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects non-numeric literal RHS") {
        auto eq = parseEquation("W[0] = X[1]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects expression RHS") {
        auto eq = parseEquation("W[0] = 1.0 + 2.0");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects pooling projections") {
        auto eq = parseEquation("W[0] += 1.0");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }
}

TEST_CASE("ScalarAssignExecutor::execute", "[executor][scalar]") {
    ScalarAssignExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Assigns scalar to indexed element") {
        auto eq = parseEquation("W[1, 2] = 3.5");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 2);
        REQUIRE(result.size(0) >= 2);
        REQUIRE(result.size(1) >= 3);
        REQUIRE_THAT(result.index({1, 2}).item<float>(),
                     Catch::Matchers::WithinRel(3.5f, 0.001f));
    }

    SECTION("Assigns to multiple indices") {
        auto eq = parseEquation("W[2, 3] = 7.0");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.size(0) >= 3);
        REQUIRE(result.size(1) >= 4);
        REQUIRE_THAT(result.index({2, 3}).item<float>(),
                     Catch::Matchers::WithinRel(7.0f, 0.001f));
    }

    SECTION("Expands tensor if needed") {
        auto eq1 = parseEquation("W[0, 0] = 1.0");
        executor.execute(eq1, env, *backend);
        env.bind("W", executor.execute(eq1, env, *backend));

        auto eq2 = parseEquation("W[5, 5] = 2.0");
        Tensor result = executor.execute(eq2, env, *backend);

        REQUIRE(result.size(0) >= 6);
        REQUIRE(result.size(1) >= 6);
    }
}

// ============================================================================
// ListLiteralExecutor Tests
// ============================================================================

TEST_CASE("ListLiteralExecutor::canExecute", "[executor][list]") {
    ListLiteralExecutor executor;
    Environment env;

    SECTION("Recognizes 1D list literal") {
        auto eq = parseEquation("X = [1.0, 2.0, 3.0]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes 2D list literal") {
        auto eq = parseEquation("X = [[1.0, 2.0], [3.0, 4.0]]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Rejects scalar RHS") {
        auto eq = parseEquation("X = 1.0");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects tensor reference RHS") {
        auto eq = parseEquation("X = Y[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }
}

TEST_CASE("ListLiteralExecutor::execute", "[executor][list]") {
    ListLiteralExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Creates 1D tensor from list") {
        auto eq = parseEquation("X = [1.0, 2.0, 3.0]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 1);
        REQUIRE(result.size(0) == 3);
        REQUIRE_THAT(result.index({0}).item<float>(),
                     Catch::Matchers::WithinRel(1.0f, 0.001f));
        REQUIRE_THAT(result.index({1}).item<float>(),
                     Catch::Matchers::WithinRel(2.0f, 0.001f));
        REQUIRE_THAT(result.index({2}).item<float>(),
                     Catch::Matchers::WithinRel(3.0f, 0.001f));
    }

    SECTION("Creates 2D tensor from nested lists") {
        auto eq = parseEquation("X = [[1.0, 2.0], [3.0, 4.0]]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 2);
        REQUIRE(result.size(0) == 2);
        REQUIRE(result.size(1) == 2);
        REQUIRE_THAT(result.index({0, 0}).item<float>(),
                     Catch::Matchers::WithinRel(1.0f, 0.001f));
        REQUIRE_THAT(result.index({1, 1}).item<float>(),
                     Catch::Matchers::WithinRel(4.0f, 0.001f));
    }

    SECTION("Creates 3D tensor") {
        auto eq = parseEquation("X = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 3);
        REQUIRE(result.size(0) == 2);
        REQUIRE(result.size(1) == 2);
        REQUIRE(result.size(2) == 2);
    }
}

// ============================================================================
// EinsumExecutor Tests
// ============================================================================

TEST_CASE("EinsumExecutor::canExecute", "[executor][einsum]") {
    EinsumExecutor executor;
    Environment env;

    // Set up test tensors
    env.bind("X", torch::randn({3, 4}));
    env.bind("Y", torch::randn({4, 5}));

    SECTION("Recognizes einsum call") {
        auto eq = parseEquation("Z[i, j] = einsum(\"ik,kj->ij\", X, Y)");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Rejects non-einsum function calls") {
        auto eq = parseEquation("Z = relu(X)");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects non-function RHS") {
        auto eq = parseEquation("Z = X[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }
}

TEST_CASE("EinsumExecutor::execute", "[executor][einsum]") {
    EinsumExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Matrix multiplication via einsum") {
        env.bind("A", torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}));
        env.bind("B", torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}}));

        auto eq = parseEquation("C[i, j] = einsum(\"ik,kj->ij\", A, B)");
        Tensor result = executor.execute(eq, env, *backend);

        // Verify shape
        REQUIRE(result.dim() == 2);
        REQUIRE(result.size(0) == 2);
        REQUIRE(result.size(1) == 2);

        // Check values: C[0,0] = 1*5 + 2*7 = 19
        REQUIRE_THAT(result.index({0, 0}).item<float>(),
                     Catch::Matchers::WithinRel(19.0f, 0.001f));
    }

    SECTION("Trace via einsum") {
        env.bind("M", torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}));

        auto eq = parseEquation("trace = einsum(\"ii->\", M)");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 0);
        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(5.0f, 0.001f)); // 1 + 4 = 5
    }
}

// ============================================================================
// IndexedProductExecutor Tests
// ============================================================================

TEST_CASE("IndexedProductExecutor::canExecute", "[executor][product]") {
    IndexedProductExecutor executor;
    Environment env;

    env.bind("A", torch::randn({3, 4}));
    env.bind("B", torch::randn({4, 5}));

    SECTION("Recognizes matrix multiplication pattern") {
        auto eq = parseEquation("C[i, j] = A[i, k] * B[k, j]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes dot product pattern") {
        env.bind("x", torch::randn({10}));
        env.bind("y", torch::randn({10}));
        auto eq = parseEquation("result = x[i] * y[i]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Rejects addition") {
        auto eq = parseEquation("C[i, j] = A[i, j] + B[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects single tensor reference") {
        auto eq = parseEquation("C[i, j] = A[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }
}

TEST_CASE("IndexedProductExecutor::execute", "[executor][product]") {
    IndexedProductExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Matrix multiplication") {
        env.bind("A", torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}));
        env.bind("B", torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}}));

        auto eq = parseEquation("C[i, j] = A[i, k] * B[k, j]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 2);
        REQUIRE_THAT(result.index({0, 0}).item<float>(),
                     Catch::Matchers::WithinRel(19.0f, 0.001f));
    }

    SECTION("Dot product") {
        env.bind("x", torch::tensor({1.0f, 2.0f, 3.0f}));
        env.bind("y", torch::tensor({4.0f, 5.0f, 6.0f}));

        auto eq = parseEquation("result = x[i] * y[i]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 0);
        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(32.0f, 0.001f)); // 1*4 + 2*5 + 3*6 = 32
    }
}

// ============================================================================
// ReductionExecutor Tests
// ============================================================================

TEST_CASE("ReductionExecutor::canExecute", "[executor][reduction]") {
    ReductionExecutor executor;
    Environment env;

    env.bind("X", torch::randn({3, 4, 5}));

    SECTION("Recognizes scalar reduction from indexed tensor") {
        auto eq = parseEquation("total = X[i, j, k]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes reduction from 1D tensor") {
        env.bind("V", torch::randn({10}));
        auto eq = parseEquation("sum = V[i]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Rejects indexed LHS") {
        auto eq = parseEquation("Y[i] = X[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects pooling projections") {
        auto eq = parseEquation("total += X[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }
}

TEST_CASE("ReductionExecutor::execute", "[executor][reduction]") {
    ReductionExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Sum 2D tensor to scalar") {
        env.bind("X", torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}));

        auto eq = parseEquation("total = X[i, j]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 0);
        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(21.0f, 0.001f)); // 1+2+3+4+5+6
    }

    SECTION("Sum 1D tensor to scalar") {
        env.bind("X", torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}));

        auto eq = parseEquation("total = X[i]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 0);
        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(10.0f, 0.001f)); // 1+2+3+4
    }
}

// ============================================================================
// PoolingExecutor Tests
// ============================================================================

TEST_CASE("PoolingExecutor::canExecute", "[executor][pooling]") {
    PoolingExecutor executor;
    Environment env;

    env.bind("X", torch::randn({4, 8}));

    SECTION("Recognizes += pooling") {
        auto eq = parseEquation("Y[i, j/2] += X[i, j]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes avg= pooling") {
        auto eq = parseEquation("Y[i, j/2] avg= X[i, j]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes max= pooling") {
        auto eq = parseEquation("Y[i, j/2] max= X[i, j]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes min= pooling") {
        auto eq = parseEquation("Y[i, j/2] min= X[i, j]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Rejects standard assignment") {
        auto eq = parseEquation("Y[i, j/2] = X[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Accepts non-strided pooling") {
        auto eq = parseEquation("Y[i, j] += X[i, j]");
        REQUIRE(executor.canExecute(eq, env));
    }
}

TEST_CASE("PoolingExecutor::execute", "[executor][pooling]") {
    PoolingExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Sum pooling with stride 2") {
        env.bind("X", torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f}}));

        auto eq = parseEquation("Y[i, j/2] += X[i, j]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 2);
        REQUIRE(result.size(1) == 2);
        REQUIRE_THAT(result.index({0, 0}).item<float>(),
                     Catch::Matchers::WithinRel(3.0f, 0.001f)); // 1+2
        REQUIRE_THAT(result.index({0, 1}).item<float>(),
                     Catch::Matchers::WithinRel(7.0f, 0.001f)); // 3+4
    }

    SECTION("Average pooling with stride 2") {
        env.bind("X", torch::tensor({{2.0f, 4.0f, 6.0f, 8.0f}}));

        auto eq = parseEquation("Y[i, j/2] avg= X[i, j]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.index({0, 0}).item<float>(),
                     Catch::Matchers::WithinRel(3.0f, 0.001f)); // (2+4)/2
        REQUIRE_THAT(result.index({0, 1}).item<float>(),
                     Catch::Matchers::WithinRel(7.0f, 0.001f)); // (6+8)/2
    }

    SECTION("Max pooling with stride 2") {
        env.bind("X", torch::tensor({{1.0f, 5.0f, 2.0f, 8.0f}}));

        auto eq = parseEquation("Y[i, j/2] max= X[i, j]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.index({0, 0}).item<float>(),
                     Catch::Matchers::WithinRel(5.0f, 0.001f)); // max(1,5)
        REQUIRE_THAT(result.index({0, 1}).item<float>(),
                     Catch::Matchers::WithinRel(8.0f, 0.001f)); // max(2,8)
    }

    SECTION("Min pooling with stride 2") {
        env.bind("X", torch::tensor({{3.0f, 1.0f, 9.0f, 4.0f}}));

        auto eq = parseEquation("Y[i, j/2] min= X[i, j]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.index({0, 0}).item<float>(),
                     Catch::Matchers::WithinRel(1.0f, 0.001f)); // min(3,1)
        REQUIRE_THAT(result.index({0, 1}).item<float>(),
                     Catch::Matchers::WithinRel(4.0f, 0.001f)); // min(9,4)
    }
}

// ============================================================================
// IdentityExecutor Tests
// ============================================================================

TEST_CASE("IdentityExecutor::canExecute", "[executor][identity]") {
    IdentityExecutor executor;
    Environment env;

    env.bind("X", torch::randn({3, 4}));

    SECTION("Recognizes identity assignment") {
        auto eq = parseEquation("Y[i, j] = X[i, j]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes identity with different variable names") {
        auto eq = parseEquation("Y[a, b, c] = X[a, b, c]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Accepts transposition (doesn't validate index order)") {
        auto eq = parseEquation("Y[i, j] = X[j, i]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Rejects reduction (handled by ReductionExecutor)") {
        auto eq = parseEquation("Y = X[i, j]");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }

    SECTION("Rejects scalar RHS") {
        auto eq = parseEquation("Y[i, j] = 1.0");
        REQUIRE_FALSE(executor.canExecute(eq, env));
    }
}

TEST_CASE("IdentityExecutor::execute", "[executor][identity]") {
    IdentityExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Copies tensor without modification") {
        Tensor original = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
        env.bind("X", original);

        auto eq = parseEquation("Y[i, j] = X[i, j]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 2);
        REQUIRE(result.size(0) == 2);
        REQUIRE(result.size(1) == 2);
        REQUIRE(torch::allclose(result, original));
    }

    SECTION("Works with 3D tensors") {
        Tensor original = torch::randn({2, 3, 4});
        env.bind("X", original);

        auto eq = parseEquation("Y[a, b, c] = X[a, b, c]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 3);
        REQUIRE(torch::allclose(result, original));
    }
}

// ============================================================================
// ExpressionExecutor Tests
// ============================================================================

TEST_CASE("ExpressionExecutor::canExecute", "[executor][expression]") {
    ExpressionExecutor executor;
    Environment env;

    env.bind("X", torch::randn({3, 4}));
    env.bind("Y", torch::randn({3, 4}));

    SECTION("Recognizes arithmetic expressions") {
        auto eq = parseEquation("Z = X[i, j] + Y[i, j]");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes function calls") {
        auto eq = parseEquation("Z = relu(X[i, j])");
        REQUIRE(executor.canExecute(eq, env));
    }

    SECTION("Recognizes complex expressions") {
        auto eq = parseEquation("Z = 2.0 * X[i, j] + relu(Y[i, j])");
        REQUIRE(executor.canExecute(eq, env));
    }
}

TEST_CASE("ExpressionExecutor::execute - arithmetic", "[executor][expression]") {
    ExpressionExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Addition") {
        env.bind("A", torch::tensor({1.0f, 2.0f, 3.0f}));
        env.bind("B", torch::tensor({4.0f, 5.0f, 6.0f}));

        auto eq = parseEquation("C[i] = A[i] + B[i]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.index({0}).item<float>(),
                     Catch::Matchers::WithinRel(5.0f, 0.001f));
        REQUIRE_THAT(result.index({1}).item<float>(),
                     Catch::Matchers::WithinRel(7.0f, 0.001f));
        REQUIRE_THAT(result.index({2}).item<float>(),
                     Catch::Matchers::WithinRel(9.0f, 0.001f));
    }

    SECTION("Subtraction") {
        env.bind("A", torch::tensor({5.0f, 10.0f, 15.0f}));
        env.bind("B", torch::tensor({1.0f, 2.0f, 3.0f}));

        auto eq = parseEquation("C[i] = A[i] - B[i]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.index({0}).item<float>(),
                     Catch::Matchers::WithinRel(4.0f, 0.001f));
    }

    SECTION("Multiplication") {
        env.bind("A", torch::tensor({2.0f, 3.0f}));
        env.bind("B", torch::tensor({4.0f, 5.0f}));

        auto eq = parseEquation("C[i] = A[i] * B[i]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.index({0}).item<float>(),
                     Catch::Matchers::WithinRel(8.0f, 0.001f));
        REQUIRE_THAT(result.index({1}).item<float>(),
                     Catch::Matchers::WithinRel(15.0f, 0.001f));
    }

    SECTION("Division") {
        env.bind("A", torch::tensor({10.0f, 20.0f}));
        env.bind("B", torch::tensor({2.0f, 4.0f}));

        auto eq = parseEquation("C[i] = A[i] / B[i]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.index({0}).item<float>(),
                     Catch::Matchers::WithinRel(5.0f, 0.001f));
        REQUIRE_THAT(result.index({1}).item<float>(),
                     Catch::Matchers::WithinRel(5.0f, 0.001f));
    }
}

TEST_CASE("ExpressionExecutor::execute - activations", "[executor][expression]") {
    ExpressionExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("ReLU on vector") {
        env.bind("X", torch::tensor({-1.0f, 0.0f, 2.0f}));

        auto eq = parseEquation("Y[i] = relu(X[i])");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 1);
        REQUIRE_THAT(result.index({0}).item<float>(),
                     Catch::Matchers::WithinRel(0.0f, 0.001f));
        REQUIRE_THAT(result.index({1}).item<float>(),
                     Catch::Matchers::WithinRel(0.0f, 0.001f));
        REQUIRE_THAT(result.index({2}).item<float>(),
                     Catch::Matchers::WithinRel(2.0f, 0.001f));
    }

    SECTION("Sigmoid on scalar") {
        env.bind("X", torch::tensor(0.0f));

        auto eq = parseEquation("Y = sigmoid(X)");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 0);
        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(0.5f, 0.001f));
    }

    SECTION("Tanh on scalar") {
        env.bind("X", torch::tensor(0.0f));

        auto eq = parseEquation("Y = tanh(X)");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE(result.dim() == 0);
        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(0.0f, 0.001f));
    }
}

TEST_CASE("ExpressionExecutor::execute - element assignment with label creation", "[executor][expression]") {
    ExpressionExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Creates labels for indexed assignment with literal") {
        auto eq = parseEquation("W[Alice] = 5.0");
        Tensor result = executor.execute(eq, env, *backend);

        int alice_idx;
        REQUIRE(env.getLabelIndex("Alice", alice_idx));
        REQUIRE_THAT(result.index({alice_idx}).item<float>(),
                     Catch::Matchers::WithinRel(5.0f, 0.001f));
    }

    SECTION("Auto-reduces tensor to scalar when LHS is scalar") {
        env.bind("X", torch::tensor({1.0f, 2.0f, 3.0f}));

        auto eq = parseEquation("total = X[i]");
        Tensor result = executor.execute(eq, env, *backend);

        // X[i] sums to 6.0 when assigned to scalar LHS
        REQUIRE(result.dim() == 0);
        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(6.0f, 0.001f));
    }
}

TEST_CASE("ExpressionExecutor::execute - edge cases", "[executor][expression]") {
    ExpressionExecutor executor;
    Environment env;
    auto backend = createBackend();

    SECTION("Parenthesized expressions") {
        env.bind("X", torch::tensor({2.0f}));
        env.bind("Y", torch::tensor({3.0f}));
        env.bind("Z", torch::tensor({4.0f}));

        auto eq = parseEquation("result = (X[i] + Y[i]) * Z[i]");
        Tensor result = executor.execute(eq, env, *backend);

        REQUIRE_THAT(result.item<float>(),
                     Catch::Matchers::WithinRel(20.0f, 0.001f)); // (2+3)*4 = 20
    }

    SECTION("Complex nested expressions") {
        env.bind("A", torch::tensor({1.0f}));
        env.bind("B", torch::tensor({2.0f}));

        auto eq = parseEquation("result = relu(A[i] - B[i]) + sigmoid(B[i])");
        Tensor result = executor.execute(eq, env, *backend);

        // relu(1-2) = 0, sigmoid(2) ≈ 0.88, total ≈ 0.88
        REQUIRE(result.item<float>() > 0.8f);
        REQUIRE(result.item<float>() < 0.9f);
    }
}

TEST_CASE("Executor priority ordering", "[executor][priority]") {
    // Verify that executors have correct priority values
    ScalarAssignExecutor scalar;
    ListLiteralExecutor list;
    EinsumExecutor einsum;
    IndexedProductExecutor product;
    ReductionExecutor reduction;
    PoolingExecutor pooling;
    IdentityExecutor identity;
    ExpressionExecutor expression;

    REQUIRE(scalar.priority() == 10);
    REQUIRE(list.priority() == 20);
    REQUIRE(einsum.priority() == 30);
    REQUIRE(product.priority() == 35);
    REQUIRE(reduction.priority() == 40);
    REQUIRE(pooling.priority() == 50);
    REQUIRE(identity.priority() == 80);
    REQUIRE(expression.priority() == 90);
}

TEST_CASE("Executor names", "[executor][metadata]") {
    // Verify executor names for debugging
    ScalarAssignExecutor scalar;
    ListLiteralExecutor list;
    EinsumExecutor einsum;
    IndexedProductExecutor product;
    ReductionExecutor reduction;
    PoolingExecutor pooling;
    IdentityExecutor identity;
    ExpressionExecutor expression;

    REQUIRE(scalar.name() == "ScalarAssignExecutor");
    REQUIRE(list.name() == "ListLiteralExecutor");
    REQUIRE(einsum.name() == "EinsumExecutor");
    REQUIRE(product.name() == "IndexedProductExecutor");
    REQUIRE(reduction.name() == "ReductionExecutor");
    REQUIRE(pooling.name() == "PoolingExecutor");
    REQUIRE(identity.name() == "IdentityExecutor");
    REQUIRE(expression.name() == "ExpressionExecutor");
}
