#include "TL/backend.hpp"

#include <torch/torch.h>
#include <iostream>

namespace tl {

class LibTorchBackend final : public TensorBackend {
public:
  LibTorchBackend() = default;
  ~LibTorchBackend() override = default;

  Tensor compute(const Equation &eq) override {
    switch (eq.kind) {
    case Equation::Kind::Constant:
      return eq.constant;
    case Equation::Kind::Identity:
      if (!eq.operands.empty()) {
        return eq.operands.front();
      }
      // Fall through to constant if provided
      if (eq.constant.defined()) return eq.constant;
      throw std::invalid_argument("Identity equation missing operand");
    case Equation::Kind::Einsum:
      return einsum(eq.einsum_spec, eq.operands);
    }
    throw std::logic_error("Unknown equation kind");
  }

  Tensor einsum(const std::string &indices,
                const std::vector<Tensor> &tensors) override {
    // Use at::einsum from LibTorch
    return torch::einsum(indices, tensors);
  }

  void learn(const Program &, const Loss &) override {
    // Phase 1 stub: learning handled at program level in future steps
  }
};

std::unique_ptr<TensorBackend> BackendFactory::create(BackendType type) {
  switch (type) {
  case BackendType::LibTorch:
    return std::make_unique<LibTorchBackend>();
  }
  throw std::invalid_argument("Unsupported backend type");
}

std::unique_ptr<TensorBackend> BackendFactory::createHybrid(
    std::unique_ptr<TensorBackend> sparse,
    std::unique_ptr<TensorBackend> dense) {
  // Phase 1: no hybrid routing. Prefer dense backend if provided.
  if (dense) return dense;
  return sparse;
}

} // namespace tl
