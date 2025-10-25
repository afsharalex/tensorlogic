#pragma once

#include "TL/AST.hpp"
#include "TL/backend.hpp"
#include "TL/Runtime/ExecutorRegistry.hpp"
#include "TL/Runtime/PreprocessorRegistry.hpp"
#include "TL/Runtime/DatalogEngine.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>

namespace tl {

// Simple runtime environment that maps tensor names to concrete Tensor values
// and stores simple Datalog facts (as Boolean relations).
class Environment {
public:
  void bind(const std::string &name, const Tensor &t);
  void bind(const TensorRef &ref, const Tensor &t);

  bool has(const std::string &name) const;
  bool has(const TensorRef &ref) const;

  const Tensor &lookup(const std::string &name) const; // throws if missing
  const Tensor &lookup(const TensorRef &ref) const;     // throws if missing

  static std::string key(const TensorRef &ref);

  // Label indexing for uppercase constants used as tensor indices
  // Returns an existing index for label or creates a new one.
  int internLabel(const std::string &label);
  // Returns true and sets outIdx if label has an assigned index.
  bool getLabelIndex(const std::string &label, int &outIdx) const;

  // Datalog fact storage helpers (Phase 1 minimal)
  bool addFact(const DatalogFact &f); // returns true if inserted new fact
  bool addFact(const std::string &relation, const std::vector<std::string> &tuple); // returns true if new
  bool hasRelation(const std::string &relation) const;
  const std::vector<std::vector<std::string>> &facts(const std::string &relation) const; // empty if missing

private:
  std::unordered_map<std::string, Tensor> tensors_;
  // Global mapping from string labels (e.g., Alice) to stable integer indices for tensor axes.
  std::unordered_map<std::string, int> labelToIndex_;
  // Map relation -> list of tuples (each tuple is a vector of constants as strings)
  std::unordered_map<std::string, std::vector<std::vector<std::string>>> datalog_;
  // For fast deduplication: per-relation set of serialized tuples
  std::unordered_map<std::string, std::unordered_set<std::string>> datalog_set_;
};

// Minimal router for Phase 1: route tensor equations to LibTorch.
class BackendRouter {
public:
  static BackendType analyze(const Statement &st);
};

// Interpreted VM (Phase 1): walk statements and execute eagerly.
class TensorLogicVM {
public:
  TensorLogicVM(std::ostream* out = &std::cout, std::ostream* err = &std::cerr);

  // Enable/disable debug logging
  void setDebug(bool enabled);
  bool debug() const;

  // Execute a full program. For Phase 1, this executes tensor equations
  // that we can interpret (currently limited to einsum calls with existing
  // tensors in the environment). Adds minimal Datalog fact/query support.
  void execute(const Program &program);

  // Access the environment (e.g., for tests or embedding)
  Environment &env() { return env_; }
  const Environment &env() const { return env_; }

private:
  void execTensorEquation(const TensorEquation &eq);
  void execQuery(const Query &q);
  void initializeExecutors();
  void initializePreprocessors();

  void debugLog(const std::string &msg) const;

  // Output streams for normal output and errors/debug
  std::ostream* output_stream_ { &std::cout };
  std::ostream* error_stream_ { &std::cerr };

  std::unique_ptr<TensorBackend> torch_;
  BackendRouter router_;
  Environment env_;
  bool debug_{false};
  PreprocessorRegistry preprocessor_registry_;
  ExecutorRegistry executor_registry_;
  DatalogEngine datalog_engine_;
};

} // namespace tl
