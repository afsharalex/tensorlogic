#include "TL/AST.hpp"
#include "TL/Parser.hpp"
#include "TL/backend.hpp"
#include "TL/vm.hpp"
#include <iostream>
#include <torch/torch.h>

/// Parses, Evaluates/Executes the given '.tl' file
void runFile(const std::string &fileName, bool debug) {
  try {
    const tl::Program prog = tl::parseFile(fileName);
    std::cout << "Parsed program: " << prog.statements.size() << " statement(s)"
              << std::endl;
    // Print a short preview
    size_t count = 0;
    for (const auto &st : prog.statements) {
      if (count++ >= 10) {
        std::cout << "..." << std::endl;
        break;
      }
      std::cout << "  - " << tl::toString(st) << std::endl;
    }

    // Execute program
    tl::TensorLogicVM vm;
    vm.setDebug(debug);
    vm.execute(prog);
    std::cout << "Executed program successfully." << std::endl;
  } catch (const tl::ParseError &e) {
    std::cerr << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Execution error: " << e.what() << std::endl;
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
  bool debug = false;
  int argi = 1;
  while (argi < argc && argv[argi][0] == '-') {
    std::string opt = argv[argi];
    if (opt == "--debug" || opt == "-d") {
      debug = true;
      ++argi;
      continue;
    }
    std::cerr << "Unknown option: " << opt << "\n";
    std::cerr << "Usage: tl [--debug|-d] <file.tl>\n";
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
    runFile(fileName, debug);
  } else {
    // TODO: Start REPL

    // Backend einsum demo
    printBackendEinsumDemo();
  }

  return 0;
}
