#pragma once

#include "TL/AST.hpp"
#include "TL/backend.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tl {

// Simple runtime environment that maps tensor names to concrete Tensor values.
class Environment {
public:
  void bind(const std::string &name, const Tensor &t);
  void bind(const TensorRef &ref, const Tensor &t);

  bool has(const std::string &name) const;
  bool has(const TensorRef &ref) const;

  const Tensor &lookup(const std::string &name) const; // throws if missing
  const Tensor &lookup(const TensorRef &ref) const;     // throws if missing

  static std::string key(const TensorRef &ref);

private:
  std::unordered_map<std::string, Tensor> tensors_;
};

// Minimal router for Phase 1: route tensor equations to LibTorch.
class BackendRouter {
public:
  static BackendType analyze(const Statement &st);
};

// Interpreted VM (Phase 1): walk statements and execute eagerly.
class TensorLogicVM {
public:
  TensorLogicVM();

  // Execute a full program. For Phase 1, this executes tensor equations
  // that we can interpret (currently limited to einsum calls with existing
  // tensors in the environment). Datalog and file ops are stubs.
  void execute(const Program &program);

  // Access the environment (e.g., for tests or embedding)
  Environment &env() { return env_; }
  const Environment &env() const { return env_; }

private:
  void execTensorEquation(const TensorEquation &eq);
  void execQuery(const Query &q);

  std::unique_ptr<TensorBackend> torch_;
  BackendRouter router_;
  Environment env_;
};

} // namespace tl
