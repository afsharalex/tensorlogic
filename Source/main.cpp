#include "TL/AST.hpp"
#include "TL/Parser.hpp"
#include "TL/backend.hpp"
#include "TL/VM.hpp"
#include "TL/Compiler.hpp"
#include "TL/BytecodeVM.hpp"
#include <iostream>
#include <optional>
#include <torch/torch.h>

struct ExecutionOptions {
  bool debug = false;
  bool compile_only = false;
  bool disassemble = false;
  bool use_bytecode = false;
};

/// Parses, Evaluates/Executes the given '.tl' file
void runFile(const std::string &fileName, const ExecutionOptions& options) {
  try {
    const tl::Program prog = tl::parseFile(fileName);
    std::cout << "Parsed program: " << prog.statements.size() << " statement(s)"
              << std::endl;

    // Print a short preview if not using bytecode
    if (!options.use_bytecode) {
      size_t count = 0;
      for (const auto &st : prog.statements) {
        if (count++ >= 10) {
          std::cout << "..." << std::endl;
          break;
        }
        std::cout << "  - " << tl::toString(st) << std::endl;
      }
    }

    if (options.use_bytecode || options.compile_only || options.disassemble) {
      // Compile to bytecode
      std::cout << "Compiling to bytecode..." << std::endl;
      tl::Compiler compiler;
      auto compile_result = compiler.compile(prog);

      if (compile_result.isErr()) {
        std::cerr << "Compilation error: " << compile_result.error().format() << std::endl;
        return;
      }

      tl::BytecodeModule module = std::move(compile_result.value());

      std::cout << "Generated " << module.instructions.size() << " instructions, "
                << module.constants.size() << " constants" << std::endl;

      if (options.disassemble) {
        std::cout << "\n" << tl::disassembleModule(module) << std::endl;
      }

      if (options.compile_only) {
        std::cout << "Compilation complete (not executing)." << std::endl;
        return;
      }

      // Execute bytecode
      auto backend = tl::BackendFactory::create(tl::BackendType::LibTorch);
      tl::Environment env;
      std::ostream* out = options.debug ? &std::cout : nullptr;
      tl::BytecodeVM vm(*backend, env, out);

      if (options.debug) {
        vm.debug(module);
      } else {
        vm.execute(module);
      }

      std::cout << "Bytecode execution complete." << std::endl;
    } else {
      // Execute with AST interpreter
      tl::TensorLogicVM vm;
      vm.setDebug(options.debug);
      vm.execute(prog);
      std::cout << "Executed program successfully." << std::endl;
    }
  } catch (const tl::ParseError &e) {
    std::cerr << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Execution error: " << e.what() << std::endl;
  }
}

std::string replHelp() {
  return "\nTensorLogic REPL - Interactive Programming Environment\n"
         "Available commands:\n"
         "  \\h        Show this help message\n"
         "  \\q        Quit the REPL\n"
         "  \\vars     List all defined tensors and relations\n"
         "  \\clear    Clear the environment (reset all variables)\n"
         "  \\debug    Toggle debug mode on/off\n"
         "\nEnter TensorLogic statements directly to execute them.\n";
}

void printEnvironment(const tl::TensorLogicVM &vm) {
  const tl::Environment &env = vm.env();

  std::cout << "\n=== Environment State ===" << std::endl;

  // Print tensors
  const auto &tensors = env.tensors();
  if (tensors.empty()) {
    std::cout << "No tensors defined." << std::endl;
  } else {
    std::cout << "\nTensors (" << tensors.size() << "):" << std::endl;
    for (const auto &[name, tensor] : tensors) {
      std::cout << "  " << name << ": shape=" << tensor.sizes()
                << " dtype=" << tensor.dtype() << std::endl;
    }
  }

  // Print Datalog relations
  const auto &relations = env.relations();
  if (!relations.empty()) {
    std::cout << "\nDatalog Relations (" << relations.size() << "):" << std::endl;
    for (const auto &[rel_name, facts] : relations) {
      std::cout << "  " << rel_name << ": " << facts.size() << " fact(s)" << std::endl;
    }
  }

  std::cout << "======================\n" << std::endl;
}

bool handleCommand(const std::string &line, std::unique_ptr<tl::TensorLogicVM> &vm, bool &shouldQuit) {
  // Check for empty input
  if (line.empty()) {
    return true;
  }

  // Handle REPL commands
  if (line[0] == '\\') {
    if (line == "\\q" || line == "\\quit") {
      shouldQuit = true;
      std::cout << "Goodbye!" << std::endl;
      return true;
    } else if (line == "\\h" || line == "\\help") {
      std::cout << replHelp();
      return true;
    } else if (line == "\\vars" || line == "\\env") {
      printEnvironment(*vm);
      return true;
    } else if (line == "\\clear" || line == "\\reset") {
      // Create a new VM to reset the environment
      const bool debugMode = vm->debug();
      vm = std::make_unique<tl::TensorLogicVM>();
      vm->setDebug(debugMode);
      std::cout << "Environment cleared." << std::endl;
      return true;
    } else if (line == "\\debug") {
      vm->setDebug(!vm->debug());
      std::cout << "Debug mode: " << (vm->debug() ? "ON" : "OFF") << std::endl;
      return true;
    } else {
      std::cerr << "Unknown command: " << line << std::endl;
      std::cout << "Type \\h for help." << std::endl;
      return false;
    }
  }

  // Parse and execute TensorLogic statement
  try {
    const tl::Program prog = tl::parseProgram(line);
    vm->execute(prog);
    return true;
  } catch (const tl::ParseError &e) {
    std::cerr << "Parse error: " << e.what() << std::endl;
    return false;
  } catch (const std::exception &e) {
    std::cerr << "Execution error: " << e.what() << std::endl;
    return false;
  }
}

/**
 * Starts the REPL (Read-Eval-Print Loop)
 */
void runRepl() {
  auto vm = std::make_unique<tl::TensorLogicVM>();
  bool shouldQuit = false;

  std::cout << "TensorLogic REPL v0.1" << std::endl;
  std::cout << "Type \\h for help, \\q to quit." << std::endl;

  while (!shouldQuit) {
    std::cout << "\n> ";

    std::string line;
    if (!std::getline(std::cin, line)) {
      // EOF or read error
      break;
    }

    handleCommand(line, vm, shouldQuit);
  }
}

/// Prints a demo of backend einsum to validate LibTorch backend
void printBackendEinsumDemo() {
  auto backend = tl::BackendFactory::create(tl::BackendType::LibTorch);
  const torch::Tensor a = torch::rand({3, 4});
  const torch::Tensor b = torch::rand({4, 5});

  const auto res = backend->einsum("ik,kj->ij", {a, b});
  std::cout << "Einsum result (3x5):\n" << res << std::endl;
}

int main(const int argc, char **argv) {

  // Parse optional flags
  ExecutionOptions options;
  int argi = 1;
  while (argi < argc && argv[argi][0] == '-') {
    std::string opt = argv[argi];
    if (opt == "--debug" || opt == "-d") {
      options.debug = true;
      ++argi;
      continue;
    } else if (opt == "--compile" || opt == "-c") {
      options.compile_only = true;
      ++argi;
      continue;
    } else if (opt == "--bytecode" || opt == "-b") {
      options.use_bytecode = true;
      ++argi;
      continue;
    } else if (opt == "--disassemble" || opt == "-D") {
      options.disassemble = true;
      ++argi;
      continue;
    }
    std::cerr << "Unknown option: " << opt << "\n";
    std::cerr << "Usage: tl [options] <file.tl>\n";
    std::cerr << "Options:\n";
    std::cerr << "  --debug, -d         Enable debug output\n";
    std::cerr << "  --bytecode, -b      Execute using bytecode VM\n";
    std::cerr << "  --compile, -c       Compile to bytecode only (don't execute)\n";
    std::cerr << "  --disassemble, -D   Show bytecode disassembly\n";
    return 1;
  }

  // Check if a file name was provided
  if (argi < argc) {
    const std::string fileName = argv[argi];

    // Check file extension is .tl
    if (fileName.size() < 3 || fileName.substr(fileName.size() - 3) != ".tl") {
      std::cout << "Invalid file extension" << std::endl;
      return 1;
    }

    // Run file
    runFile(fileName, options);
  } else {
    // Start REPL if no file provided
    runRepl();
  }

  return 0;
}
