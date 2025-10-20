#include "TL/AST.hpp"
#include "TL/Parser.hpp"
#include "TL/backend.hpp"
#include <iostream>
#include <torch/torch.h>

/// Parses, Evaluates/Executes the given '.tl' file (parse-only for now)
void runFile(const std::string &fileName) {
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
  } catch (const tl::ParseError &e) {
    std::cerr << e.what() << std::endl;
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

  // Check if a file name was provided
  if (argc > 1) {
    // Get file name
    const std::string fileName = argv[1];

    // Check file extension is .tl
    if (fileName.size() < 3 || fileName.substr(fileName.size() - 3) != ".tl") {
      std::cout << "Invalid file extension" << std::endl;
      return 1;
    }

    // Run file
    runFile(fileName);
  } else {
    // TODO: Start REPL

    // Backend einsum demo
    printBackendEinsumDemo();
  }

  return 0;
}
