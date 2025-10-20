#pragma once

#include "TL/core.hpp"

#include <memory>

namespace tl {

enum class BackendType { LibTorch };

class TensorBackend {
public:
  virtual ~TensorBackend() = default;

  // Execute a tensor equation
  virtual Tensor compute(const Equation &eq) = 0;

  // Einstein summation interface
  virtual Tensor einsum(const std::string &indices,
                        const std::vector<Tensor> &tensors) = 0;

  // Learning API (no-op for now)
  virtual void learn(const Program &prog, const Loss &loss) = 0;
};

class BackendFactory {
public:
  static std::unique_ptr<TensorBackend> create(BackendType type);

  static std::unique_ptr<TensorBackend>
  createHybrid(std::unique_ptr<TensorBackend> sparse,
               std::unique_ptr<TensorBackend> dense);
};

} // namespace tl
