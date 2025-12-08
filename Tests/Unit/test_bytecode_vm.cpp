
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "TL/Bytecode.hpp"
#include "TL/BytecodeVM.hpp"
#include "TL/Compiler.hpp"
#include "TL/Parser.hpp"
#include "TL/backend.hpp"
#include "catch2/catch_test_macros.hpp"

#include <sstream>

using namespace tl;

// Helper to create a backend instance for tests
static std::unique_ptr<TensorBackend> createBackend() {
  return BackendFactory::create(BackendType::LibTorch);
}

TEST_CASE("VM Load and Store Constants", "[bytecode][bytecode_vm]") {
  auto backend = createBackend();

  Environment env;
  std::ostringstream out;
  BytecodeVM vm{*backend, env, &out};

  BytecodeModule module{};

  // Add a scalar constant to the constant pool
  size_t const_idx = module.addConstant(Constant(torch::tensor(42.0f)));
  size_t var_idx = module.addString("x");

  // Create instructions: LOAD_CONST, STORE_VAR, HALT
  module.instructions.push_back(
      Instruction{OpCode::LOAD_CONST, 0, static_cast<uint32_t>(const_idx)});
  module.instructions.push_back(
      Instruction{OpCode::STORE_VAR, 0, static_cast<uint32_t>(var_idx)});
  module.instructions.push_back(Instruction{OpCode::HALT, uint16_t(0), uint16_t(0)});

  // Execute the bytecode
  vm.execute(module);

  // Verify the variable was stored correctly
  REQUIRE(env.has("x"));
  auto x = env.lookup("x");
  REQUIRE(x.dim() == 0); // Scalar tensor
  REQUIRE(x.item<float>() == 42.0f);
}

TEST_CASE("VM Load Multiple Constants", "[bytecode][bytecode_vm]") {
  auto backend = createBackend();

  Environment env;
  std::ostringstream out;
  BytecodeVM vm{*backend, env, &out};

  BytecodeModule module{};

  // Add multiple constants
  size_t const1_idx = module.addConstant(Constant(torch::tensor(10.0f)));
  size_t const2_idx = module.addConstant(Constant(torch::tensor(20.0f)));
  size_t const3_idx = module.addConstant(Constant(torch::tensor(30.0f)));

  size_t var1_idx = module.addString("a");
  size_t var2_idx = module.addString("b");
  size_t var3_idx = module.addString("c");

  // Load and store three different constants
  module.instructions.push_back(
      Instruction{OpCode::LOAD_CONST, 0, static_cast<uint32_t>(const1_idx)});
  module.instructions.push_back(
      Instruction{OpCode::STORE_VAR, 0, static_cast<uint32_t>(var1_idx)});

  module.instructions.push_back(
      Instruction{OpCode::LOAD_CONST, 1, static_cast<uint32_t>(const2_idx)});
  module.instructions.push_back(
      Instruction{OpCode::STORE_VAR, 1, static_cast<uint32_t>(var2_idx)});

  module.instructions.push_back(
      Instruction{OpCode::LOAD_CONST, 2, static_cast<uint32_t>(const3_idx)});
  module.instructions.push_back(
      Instruction{OpCode::STORE_VAR, 2, static_cast<uint32_t>(var3_idx)});

  module.instructions.push_back(Instruction{OpCode::HALT, uint16_t(0), uint16_t(0)});

  // Execute
  vm.execute(module);

  // Verify all three variables
  REQUIRE(env.has("a"));
  REQUIRE(env.has("b"));
  REQUIRE(env.has("c"));

  REQUIRE(env.lookup("a").item<float>() == 10.0f);
  REQUIRE(env.lookup("b").item<float>() == 20.0f);
  REQUIRE(env.lookup("c").item<float>() == 30.0f);
}

TEST_CASE("VM Debug Prints", "[bytecode][bytecode_vm]") {
  auto backend = createBackend();

  Environment env;
  std::ostringstream out;
  BytecodeVM vm{*backend, env, &out};

  BytecodeModule module{};

  // Add a constant and variable for a more realistic test
  size_t const_idx = module.addConstant(Constant(torch::tensor(99.0f)));
  size_t var_idx = module.addString("test_var");

  // Manually add a few instructions
  module.instructions.push_back(
      Instruction{OpCode::LOAD_CONST, 0, static_cast<uint32_t>(const_idx)});
  module.instructions.push_back(
      Instruction{OpCode::STORE_VAR, 0, static_cast<uint32_t>(var_idx)});
  module.instructions.push_back(Instruction{OpCode::HALT, uint16_t(0), uint16_t(0)});

  vm.debug(module);

  std::string output = out.str();
  REQUIRE(output.find("BytecodeVM Execution") != std::string::npos);
  REQUIRE(output.find("Instructions:") != std::string::npos);
  REQUIRE(output.find("LOAD_CONST") != std::string::npos);
  REQUIRE(output.find("STORE_VAR") != std::string::npos);
  REQUIRE(output.find("HALT") != std::string::npos);
}

TEST_CASE("VM Debug Print Instruction", "[bytecode][bytecode_vm]") {
  auto backend = createBackend();

  Environment env;
  std::ostringstream out;
  BytecodeVM vm{*backend, env, &out};

  BytecodeModule module{};

  // Compute 2 + 2 and print the result
  size_t const_2_idx = module.addConstant(Constant(torch::tensor(2.0f)));
  size_t result_str_idx = module.addString("result");

  // Load first constant (2.0) into register 0
  module.instructions.push_back(
      Instruction{OpCode::LOAD_CONST, 0, static_cast<uint32_t>(const_2_idx)});

  // Load second constant (2.0) into register 1
  module.instructions.push_back(
      Instruction{OpCode::LOAD_CONST, 1, static_cast<uint32_t>(const_2_idx)});

  // Add them: reg2 = reg0 + reg1
  module.instructions.push_back(
      Instruction{OpCode::ELEMENTWISE_ADD, uint16_t(2), uint16_t(0), uint16_t(1)});

  // Store result to variable for verification
  module.instructions.push_back(
      Instruction{OpCode::STORE_VAR, 2, static_cast<uint32_t>(result_str_idx)});

  // Debug print the result in register 2
  module.instructions.push_back(
      Instruction{OpCode::DEBUG_PRINT, uint16_t(2), uint16_t(0)});

  // Halt
  module.instructions.push_back(Instruction{OpCode::HALT, uint16_t(0), uint16_t(0)});

  // Execute the bytecode
  vm.execute(module);

  // Verify the computation was correct
  REQUIRE(env.has("result"));
  auto result = env.lookup("result");
  REQUIRE(result.item<float>() == 4.0f);

  // Verify the debug print output contains the value
  std::string output = out.str();
  REQUIRE(output.find("4") != std::string::npos);
  REQUIRE(output.find("DEBUG: r2") != std::string::npos);  // Should show register number
}
