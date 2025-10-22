#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

namespace tl {

// Alias our Tensor type to LibTorch's tensor for Phase 1
using Tensor = torch::Tensor;

// Minimal placeholders to unblock backend interface. These will be expanded
// as the parser/AST stabilizes.
struct Equation {
  enum class Kind {
    Constant,
    Identity,
    Einsum
  } kind = Kind::Identity;

  // Payloads for different kinds
  Tensor constant;                       // For Constant
  std::string einsum_spec;               // For Einsum
  std::vector<Tensor> operands;          // For Identity/Einsum
};

struct Program;

struct Loss {
  // Placeholder loss representation
};

} // namespace tl
