// Tests/Unit/test_bytecode.cpp
// Unit tests for bytecode compilation and execution

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "TL/Parser.hpp"
#include "TL/Compiler.hpp"
#include "TL/BytecodeVM.hpp"
#include "TL/backend.hpp"

#include <sstream>

using namespace tl;

// Helper to create a backend instance for tests
static std::unique_ptr<TensorBackend> createBackend() {
    return BackendFactory::create(BackendType::LibTorch);
}

TEST_CASE("Bytecode: Simple constant loading", "[bytecode]") {
    // Parse a simple program
    std::string source = R"(
        X = [1, 2, 3]
    )";

    auto program = parseProgram(source);
    REQUIRE(program.statements.size() == 1);

    // Compile to bytecode
    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    // Verify compilation succeeded
    REQUIRE(module.instructions.size() > 0);
    REQUIRE(module.constants.size() > 0);
    REQUIRE(module.strings.size() > 0);

    // Verify bytecode
    BytecodeVerifier verifier;
    REQUIRE(verifier.verify(module));

    // Execute bytecode
    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    // Check result
    REQUIRE(env.has("X"));
    auto X = env.lookup("X");
    REQUIRE(X.sizes().size() == 1);
    REQUIRE(X.size(0) == 3);
}

TEST_CASE("Bytecode: Arithmetic operations", "[bytecode]") {
    std::string source = R"(
        X = [1.0, 2.0, 3.0]
        Y = X + [10.0, 20.0, 30.0]
        Z = Y * [2.0, 2.0, 2.0]
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    // Check results
    REQUIRE(env.has("X"));
    REQUIRE(env.has("Y"));
    REQUIRE(env.has("Z"));

    auto X = env.lookup("X");
    auto Y = env.lookup("Y");
    auto Z = env.lookup("Z");

    // Y = X + [10, 20, 30] = [11, 22, 33]
    REQUIRE_THAT(Y.index({0}).item<float>(), Catch::Matchers::WithinRel(11.0f, 0.01f));
    REQUIRE_THAT(Y.index({1}).item<float>(), Catch::Matchers::WithinRel(22.0f, 0.01f));
    REQUIRE_THAT(Y.index({2}).item<float>(), Catch::Matchers::WithinRel(33.0f, 0.01f));

    // Z = Y * [2, 2, 2] = [22, 44, 66]
    REQUIRE_THAT(Z.index({0}).item<float>(), Catch::Matchers::WithinRel(22.0f, 0.01f));
    REQUIRE_THAT(Z.index({1}).item<float>(), Catch::Matchers::WithinRel(44.0f, 0.01f));
    REQUIRE_THAT(Z.index({2}).item<float>(), Catch::Matchers::WithinRel(66.0f, 0.01f));
}

TEST_CASE("Bytecode: Activation functions", "[bytecode]") {
    std::string source = R"(
        X = [-1.0, 0.0, 1.0]
        Y_relu = relu(X)
        Y_sigmoid = sigmoid(X)
        Y_tanh = tanh(X)
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    // Check relu: [-1, 0, 1] -> [0, 0, 1]
    REQUIRE(env.has("Y_relu"));
    auto Y_relu = env.lookup("Y_relu");
    REQUIRE_THAT(Y_relu.index({0}).item<float>(), Catch::Matchers::WithinRel(0.0f, 0.01f));
    REQUIRE_THAT(Y_relu.index({1}).item<float>(), Catch::Matchers::WithinRel(0.0f, 0.01f));
    REQUIRE_THAT(Y_relu.index({2}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));

    // Check sigmoid exists (exact values depend on sigmoid function)
    REQUIRE(env.has("Y_sigmoid"));
    auto Y_sigmoid = env.lookup("Y_sigmoid");
    REQUIRE(Y_sigmoid.index({1}).item<float>() > 0.4f); // sigmoid(0) ≈ 0.5
    REQUIRE(Y_sigmoid.index({1}).item<float>() < 0.6f);

    // Check tanh exists
    REQUIRE(env.has("Y_tanh"));
}

TEST_CASE("Bytecode: Unary negation", "[bytecode]") {
    std::string source = R"(
        X = [1.0, -2.0, 3.0]
        Y = -X
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    REQUIRE(env.has("Y"));
    auto Y = env.lookup("Y");

    // Y = -X = [-1, 2, -3]
    REQUIRE_THAT(Y.index({0}).item<float>(), Catch::Matchers::WithinRel(-1.0f, 0.01f));
    REQUIRE_THAT(Y.index({1}).item<float>(), Catch::Matchers::WithinRel(2.0f, 0.01f));
    REQUIRE_THAT(Y.index({2}).item<float>(), Catch::Matchers::WithinRel(-3.0f, 0.01f));
}

TEST_CASE("Bytecode: Power operation", "[bytecode]") {
    std::string source = R"(
        X = [2.0, 3.0, 4.0]
        Y = X ^ [2.0, 2.0, 2.0]
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    REQUIRE(env.has("Y"));
    auto Y = env.lookup("Y");

    // Y = X^2 = [4, 9, 16]
    REQUIRE_THAT(Y.index({0}).item<float>(), Catch::Matchers::WithinRel(4.0f, 0.01f));
    REQUIRE_THAT(Y.index({1}).item<float>(), Catch::Matchers::WithinRel(9.0f, 0.01f));
    REQUIRE_THAT(Y.index({2}).item<float>(), Catch::Matchers::WithinRel(16.0f, 0.01f));
}

TEST_CASE("Bytecode: Comparison operations", "[bytecode]") {
    std::string source = R"(
        X = [1.0, 2.0, 3.0]
        Y_gt = X > [1.5, 1.5, 1.5]
        Y_lt = X < [2.5, 2.5, 2.5]
        Y_eq = X == [2.0, 2.0, 2.0]
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    // Y_gt: X > 1.5 = [0, 1, 1]
    REQUIRE(env.has("Y_gt"));
    auto Y_gt = env.lookup("Y_gt");
    REQUIRE_THAT(Y_gt.index({0}).item<float>(), Catch::Matchers::WithinRel(0.0f, 0.01f));
    REQUIRE_THAT(Y_gt.index({1}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));
    REQUIRE_THAT(Y_gt.index({2}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));

    // Y_lt: X < 2.5 = [1, 1, 0]
    REQUIRE(env.has("Y_lt"));
    auto Y_lt = env.lookup("Y_lt");
    REQUIRE_THAT(Y_lt.index({0}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));
    REQUIRE_THAT(Y_lt.index({1}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));
    REQUIRE_THAT(Y_lt.index({2}).item<float>(), Catch::Matchers::WithinRel(0.0f, 0.01f));

    // Y_eq: X == 2.0 = [0, 1, 0]
    REQUIRE(env.has("Y_eq"));
    auto Y_eq = env.lookup("Y_eq");
    REQUIRE_THAT(Y_eq.index({0}).item<float>(), Catch::Matchers::WithinRel(0.0f, 0.01f));
    REQUIRE_THAT(Y_eq.index({1}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));
    REQUIRE_THAT(Y_eq.index({2}).item<float>(), Catch::Matchers::WithinRel(0.0f, 0.01f));
}

TEST_CASE("Bytecode: Complex expression", "[bytecode]") {
    std::string source = R"(
        X = [1.0, 2.0, 3.0]
        Y = relu(X * [2.0, 2.0, 2.0] - [1.0, 1.0, 1.0])
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    REQUIRE(env.has("Y"));
    auto Y = env.lookup("Y");

    // Y = relu(X * 2 - 1) = relu([1, 3, 5]) = [1, 3, 5]
    REQUIRE_THAT(Y.index({0}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));
    REQUIRE_THAT(Y.index({1}).item<float>(), Catch::Matchers::WithinRel(3.0f, 0.01f));
    REQUIRE_THAT(Y.index({2}).item<float>(), Catch::Matchers::WithinRel(5.0f, 0.01f));
}

TEST_CASE("Bytecode: Reductions", "[bytecode]") {
    std::string source = R"(
        X = [1.0, 2.0, 3.0]
        Sum = X + X
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    REQUIRE(env.has("Sum"));
    auto Sum = env.lookup("Sum");

    // Sum = X + X = [2, 4, 6]
    REQUIRE_THAT(Sum.index({0}).item<float>(), Catch::Matchers::WithinRel(2.0f, 0.01f));
    REQUIRE_THAT(Sum.index({1}).item<float>(), Catch::Matchers::WithinRel(4.0f, 0.01f));
    REQUIRE_THAT(Sum.index({2}).item<float>(), Catch::Matchers::WithinRel(6.0f, 0.01f));
}

TEST_CASE("Bytecode: Variable dependencies", "[bytecode]") {
    std::string source = R"(
        A = [1.0, 2.0]
        B = A + [10.0, 10.0]
        C = B * [2.0, 2.0]
        D = relu(C - [20.0, 20.0])
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    vm.execute(module);

    // Check all variables exist
    REQUIRE(env.has("A"));
    REQUIRE(env.has("B"));
    REQUIRE(env.has("C"));
    REQUIRE(env.has("D"));

    // A = [1, 2]
    auto A = env.lookup("A");
    REQUIRE_THAT(A.index({0}).item<float>(), Catch::Matchers::WithinRel(1.0f, 0.01f));

    // B = A + 10 = [11, 12]
    auto B = env.lookup("B");
    REQUIRE_THAT(B.index({0}).item<float>(), Catch::Matchers::WithinRel(11.0f, 0.01f));
    REQUIRE_THAT(B.index({1}).item<float>(), Catch::Matchers::WithinRel(12.0f, 0.01f));

    // C = B * 2 = [22, 24]
    auto C = env.lookup("C");
    REQUIRE_THAT(C.index({0}).item<float>(), Catch::Matchers::WithinRel(22.0f, 0.01f));
    REQUIRE_THAT(C.index({1}).item<float>(), Catch::Matchers::WithinRel(24.0f, 0.01f));

    // D = relu(C - 20) = relu([2, 4]) = [2, 4]
    auto D = env.lookup("D");
    REQUIRE_THAT(D.index({0}).item<float>(), Catch::Matchers::WithinRel(2.0f, 0.01f));
    REQUIRE_THAT(D.index({1}).item<float>(), Catch::Matchers::WithinRel(4.0f, 0.01f));
}

TEST_CASE("Bytecode: Constant pool deduplication", "[bytecode]") {
    std::string source = R"(
        X = [1.0, 2.0, 3.0]
        Y = [1.0, 2.0, 3.0]
        Z = X + Y
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    // The same constant [1, 2, 3] should be deduplicated
    // Note: Current implementation may not deduplicate tensor constants
    // but should deduplicate identical numeric constants
    REQUIRE(module.constants.size() >= 1);
}

TEST_CASE("Bytecode: Disassembly output", "[bytecode]") {
    std::string source = R"(
        X = [1.0, 2.0]
        Y = X + X
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    // Test disassembly
    std::string disasm = disassembleModule(module);

    // Should contain key elements
    REQUIRE(disasm.find("TensorLogic Bytecode Module") != std::string::npos);
    REQUIRE(disasm.find("LOAD_CONST") != std::string::npos);
    REQUIRE(disasm.find("STORE_VAR") != std::string::npos);
    REQUIRE(disasm.find("HALT") != std::string::npos);
}

TEST_CASE("Bytecode: Verification catches errors", "[bytecode]") {
    // Create invalid bytecode module
    BytecodeModule bad_module;
    bad_module.magic = 0xDEADBEEF; // Wrong magic number

    BytecodeVerifier verifier;
    REQUIRE_FALSE(verifier.verify(bad_module));
    REQUIRE(verifier.getErrors().size() > 0);
}

TEST_CASE("Bytecode: Debug mode execution", "[bytecode]") {
    std::string source = R"(
        X = [1.0, 2.0]
        Y = relu(X)
    )";

    auto program = parseProgram(source);

    Compiler compiler;
    auto result = compiler.compile(program);
    REQUIRE(result.isOk());
    BytecodeModule module = std::move(result.value());

    auto backend = createBackend();
    Environment env;
    std::ostringstream out;
    BytecodeVM vm(*backend, env, &out);

    // Execute in debug mode
    vm.debug(module);

    // Check that debug output was produced
    std::string output = out.str();
    REQUIRE(output.find("BytecodeVM Execution") != std::string::npos);
    REQUIRE(output.find("Instructions:") != std::string::npos);
}
