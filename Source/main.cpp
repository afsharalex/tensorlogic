#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "TL/Parser.hpp"
#include "TL/AST.hpp"

/// Parses, Evaluates/Executes the given '.tl' file (parse-only for now)
void runFile(const std::string &fileName) {
  try {
    const tl::Program prog = tl::parseFile(fileName);
    std::cout << "Parsed program: " << prog.statements.size() << " statement(s)" << std::endl;
    // Print a short preview
    size_t count = 0;
    for (const auto &st : prog.statements) {
      if (count++ >= 10) { std::cout << "..." << std::endl; break; }
      std::cout << "  - " << tl::toString(st) << std::endl;
    }
  } catch (const tl::ParseError &e) {
    std::cerr << e.what() << std::endl;
  }
}

/// Prints a demo of tensors
void printTensorsDemo(const torch::Tensor &/*tensor*/) {
  // Demo function to print tensors, check if libtorch is working
  const torch::Tensor x = torch::rand({3, 3});
  const torch::Tensor y = torch::rand({3, 3});
  const torch::Tensor z = torch::rand({3, 3});
  std::cout << x << std::endl;
  std::cout << y << std::endl;
  std::cout << z << std::endl;
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

    // Print demo tensors
    printTensorsDemo(torch::rand({3, 3}));
  }

  return 0;
}
