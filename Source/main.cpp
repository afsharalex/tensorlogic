#include <iostream>
#include <torch/torch.h>

// TODO: Implement this; for now just print file name
/// Parses, Evaluates/Executes the given '.tl' file
void runFile(const std::string &fileName) {
  std::cout << "File name: " << fileName << std::endl;
}

/// Prints a demo of tensors
void printTensorsDemo(const torch::Tensor &tensor) {
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
    if (fileName.substr(fileName.size() - 3) != ".tl") {
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
