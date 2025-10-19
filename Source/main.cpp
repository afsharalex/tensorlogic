#include <iostream>
#include <torch/torch.h>

int main() {
  torch::Tensor x = torch::rand({3, 3});
  torch::Tensor y = torch::rand({3, 3});
  torch::Tensor z = torch::rand({3, 3});
  std::cout << x << std::endl;
  std::cout << y << std::endl;
  std::cout << z << std::endl;
  return 0;
}
