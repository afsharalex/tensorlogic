#include "TL/vm.hpp"
#include "TL/Runtime/Executors/ScalarAssignExecutor.hpp"
#include "TL/Runtime/Executors/ListLiteralExecutor.hpp"
#include "TL/Runtime/Executors/EinsumExecutor.hpp"
#include "TL/Runtime/Executors/IndexedProductExecutor.hpp"
#include "TL/Runtime/Executors/ReductionExecutor.hpp"
#include "TL/Runtime/Executors/GuardedClauseExecutor.hpp"
#include "TL/Runtime/Executors/PoolingExecutor.hpp"
#include "TL/Runtime/Executors/IdentityExecutor.hpp"
#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include "TL/Runtime/Preprocessors/VirtualIndexPreprocessor.hpp"
#include "TL/Runtime/DatalogEngine.hpp"

#include <stdexcept>
#include <torch/torch.h>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cctype>
#include <functional>
#include <fstream>
#include <filesystem>

namespace tl {

// -------- Environment --------

void Environment::bind(const std::string &name, const Tensor &t) {
  tensors_[name] = t;
}

void Environment::bind(const TensorRef &ref, const Tensor &t) {
  bind(key(ref), t);
}

bool Environment::has(const std::string &name) const {
  return tensors_.find(name) != tensors_.end();
}

bool Environment::has(const TensorRef &ref) const { return has(key(ref)); }

const Tensor &Environment::lookup(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("Environment: tensor not found: " + name);
  }
  return it->second;
}

const Tensor &Environment::lookup(const TensorRef &ref) const {
  return lookup(key(ref));
}

std::string Environment::key(const TensorRef &ref) { return ref.name.name; }

int Environment::internLabel(const std::string &label) {
  auto it = labelToIndex_.find(label);
  if (it != labelToIndex_.end()) return it->second;
  int idx = static_cast<int>(labelToIndex_.size());
  labelToIndex_[label] = idx;
  return idx;
}

bool Environment::getLabelIndex(const std::string &label, int &outIdx) const {
  auto it = labelToIndex_.find(label);
  if (it == labelToIndex_.end()) return false;
  outIdx = it->second;
  return true;
}

bool Environment::addFact(const std::string &relation, const std::vector<std::string> &tuple) {
  // Serialize tuple with unit separator for unique key
  std::string key;
  for (size_t i = 0; i < tuple.size(); ++i) {
    if (i) key.push_back('\x1F');
    key += tuple[i];
  }
  auto &setRef = datalog_set_[relation];
  auto inserted = setRef.insert(key).second;
  if (inserted) {
    datalog_[relation].push_back(tuple);
  }
  return inserted;
}

bool Environment::addFact(const DatalogFact &f) {
  std::vector<std::string> tuple;
  tuple.reserve(f.constants.size());
  for (const auto &c : f.constants) tuple.push_back(c.text);
  return addFact(f.relation.name, tuple);
}

bool Environment::hasRelation(const std::string &relation) const {
  return datalog_.find(relation) != datalog_.end();
}

const std::vector<std::vector<std::string>> &Environment::facts(const std::string &relation) const {
  static const std::vector<std::vector<std::string>> kEmpty;
  auto it = datalog_.find(relation);
  if (it == datalog_.end()) return kEmpty;
  return it->second;
}

// -------- BackendRouter --------

BackendType BackendRouter::analyze(const Statement &st) {
  // Phase 1: Tensor equations go to LibTorch. Others ignored for now.
  if (std::holds_alternative<TensorEquation>(st)) {
    return BackendType::LibTorch;
  }
  return BackendType::LibTorch; // default
}

// -------- TensorLogicVM --------

TensorLogicVM::TensorLogicVM(std::ostream* out, std::ostream* err)
  : output_stream_(out), error_stream_(err), datalog_engine_(env_, out) {
  torch_ = BackendFactory::create(BackendType::LibTorch);
  if (const char* env = std::getenv("TL_DEBUG")) {
    std::string v = env;
    for (auto &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "yes" || v == "on") {
      debug_ = true;
      datalog_engine_.setDebug(true);
    }
  }
  initializePreprocessors();
  initializeExecutors();
}

void TensorLogicVM::initializePreprocessors() {
  // Register preprocessors in order of priority (lower number = processed first)
  preprocessor_registry_.registerPreprocessor(std::make_unique<VirtualIndexPreprocessor>()); // 5

  // Pass debug settings to registry
  preprocessor_registry_.setDebug(debug_);
  preprocessor_registry_.setErrOut(error_stream_);
}

void TensorLogicVM::initializeExecutors() {
  // Register executors in order of priority (lower number = higher priority)
  executor_registry_.registerExecutor(std::make_unique<ScalarAssignExecutor>());       // 10
  executor_registry_.registerExecutor(std::make_unique<ListLiteralExecutor>());        // 20
  executor_registry_.registerExecutor(std::make_unique<EinsumExecutor>());             // 30
  executor_registry_.registerExecutor(std::make_unique<IndexedProductExecutor>());     // 35
  executor_registry_.registerExecutor(std::make_unique<ReductionExecutor>());          // 40
  executor_registry_.registerExecutor(std::make_unique<GuardedClauseExecutor>());      // 50
  executor_registry_.registerExecutor(std::make_unique<PoolingExecutor>());            // 50
  executor_registry_.registerExecutor(std::make_unique<IdentityExecutor>());           // 80
  executor_registry_.registerExecutor(std::make_unique<ExpressionExecutor>());         // 90
  // VirtualIndexExecutor removed - now handled by VirtualIndexPreprocessor!

  // Pass debug settings to registry
  executor_registry_.setDebug(debug_);
  executor_registry_.setErrOut(error_stream_);
}

void TensorLogicVM::setDebug(bool enabled) {
  debug_ = enabled;
  preprocessor_registry_.setDebug(enabled);
  executor_registry_.setDebug(enabled);
  datalog_engine_.setDebug(enabled);
}
bool TensorLogicVM::debug() const { return debug_; }
void TensorLogicVM::debugLog(const std::string &msg) const {
  if (debug_) {
    (*error_stream_) << "[VM] " << msg << std::endl;
  }
}

void TensorLogicVM::execute(const Program &program) {
  for (size_t i = 0; i < program.statements.size(); ++i) {
    const auto &st = program.statements[i];
    if (debug_) {
      debugLog("Stmt " + std::to_string(i) + ": " + toString(st));
    }

    // FIRST: Preprocess statement (may expand into multiple statements)
    auto preprocessed = preprocessor_registry_.preprocess(st, env_);

    // THEN: Execute each preprocessed statement
    for (const auto &preprocessed_st : preprocessed) {
      const BackendType be = router_.analyze(preprocessed_st);
      if (debug_ && preprocessed.size() > 1) {
        debugLog("  Preprocessed: " + toString(preprocessed_st));
      }

      if (std::holds_alternative<TensorEquation>(preprocessed_st)) {
        execTensorEquation(std::get<TensorEquation>(preprocessed_st));
      } else if (std::holds_alternative<Query>(preprocessed_st)) {
        // Ensure closure is up-to-date before answering queries
        datalog_engine_.saturate();
        execQuery(std::get<Query>(preprocessed_st));
      } else if (std::holds_alternative<DatalogFact>(preprocessed_st)) {
        datalog_engine_.addFact(std::get<DatalogFact>(preprocessed_st));
      } else if (std::holds_alternative<DatalogRule>(preprocessed_st)) {
        datalog_engine_.addRule(std::get<DatalogRule>(preprocessed_st));
      } else if (std::holds_alternative<FileOperation>(preprocessed_st)) {
        const auto &fo = std::get<FileOperation>(preprocessed_st);
        auto resolvePath = [](const std::string &p)->std::filesystem::path {
        std::filesystem::path path(p);
        if (path.is_absolute()) return path;
        // Try as-is relative to CWD
        const std::filesystem::path cwd = std::filesystem::current_path();
        std::filesystem::path candidate = cwd / path;
        if (std::filesystem::exists(candidate)) return candidate;
        // TODO: Should we throw here?
        // Fall back to as-is
        return candidate;
      };

      auto readTensorFromFile = [&](const std::string &p)->Tensor {
        const std::filesystem::path rp = resolvePath(p);
        std::ifstream ifs(rp);
        if (!ifs) throw std::runtime_error("Cannot open file for reading: " + rp.string());
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(ifs, line)) {
          // Trim CR and whitespace at both ends
          while (!line.empty() && (line.back()=='\r' || line.back()=='\n' || line.back()==' ' || line.back()=='\t')) line.pop_back();
          size_t start = 0; while (start < line.size() && (line[start]==' ' || line[start]=='\t')) ++start;
          if (start > 0) line = line.substr(start);
          if (line.empty()) continue;
          lines.push_back(line);
        }
        if (lines.empty()) {
          return torch::zeros({0});
        }
        bool hasComma = false;
        for (const auto &ln : lines) { if (ln.find(',') != std::string::npos) { hasComma = true; break; } }
        if (hasComma) {
          // Parse as 2D CSV
          std::vector<float> values;
          size_t cols = 0;
          for (const auto &ln : lines) {
            std::vector<float> row;
            size_t pos = 0;
            while (pos <= ln.size()) {
              size_t comma = ln.find(',', pos);
              const std::string tok = (comma == std::string::npos) ? ln.substr(pos) : ln.substr(pos, comma - pos);
              if (!tok.empty()) {
                row.push_back(static_cast<float>(std::stod(tok)));
              } else {
                row.push_back(0.0f);
              }
              if (comma == std::string::npos) break;
              pos = comma + 1;
            }
            if (cols == 0) cols = row.size();
            if (row.size() != cols) throw std::runtime_error("CSV has inconsistent number of columns in: " + rp.string());
            values.insert(values.end(), row.begin(), row.end());
          }
          const int64_t rows = static_cast<int64_t>(lines.size());
          const int64_t c = static_cast<int64_t>(cols);
          torch::Tensor t = torch::from_blob(values.data(), {rows, c}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
          return t;
        } else {
          // Treat as 1D: one number per non-empty line
          std::vector<float> values;
          values.reserve(lines.size());
          for (const auto &ln : lines) {
            values.push_back(static_cast<float>(std::stod(ln)));
          }
          const int64_t n = static_cast<int64_t>(values.size());
          torch::Tensor t = torch::from_blob(values.data(), {n}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
          return t;
        }
      };

      auto writeTensorToFile = [&](const std::string &p, const Tensor &t) {
        std::filesystem::path rp = resolvePath(p);
        // Ensure directory exists
        std::filesystem::path parent = rp.parent_path();
        if (!parent.empty()) {
          std::error_code ec; std::filesystem::create_directories(parent, ec);
        }
        std::ofstream ofs(rp);
        if (!ofs) throw std::runtime_error("Cannot open file for writing: " + rp.string());
        const torch::Tensor contig = t.contiguous();
        if (contig.dim() == 0) {
          double v = 0.0; try { v = contig.item<double>(); } catch (...) { v = contig.item<float>(); }
          ofs << v << "\n";
          return;
        }
        if (contig.dim() == 1) {
          const int64_t n = contig.size(0);
          for (int64_t i = 0; i < n; ++i) {
            double v = 0.0; try { v = contig[i].item<double>(); } catch (...) { v = contig[i].item<float>(); }
            ofs << v;
            if (i + 1 < n) ofs << "\n";
          }
          return;
        }
        if (contig.dim() == 2) {
          const int64_t r = contig.size(0);
          const int64_t c = contig.size(1);
          for (int64_t i = 0; i < r; ++i) {
            for (int64_t j = 0; j < c; ++j) {
              double v = 0.0; try { v = contig[i][j].item<double>(); } catch (...) { v = contig[i][j].item<float>(); }
              if (j) ofs << ",";
              ofs << v;
            }
            if (i + 1 < r) ofs << "\n";
          }
          return;
        }
        // Higher dimensions: write flattened, one value per line
        const int64_t n = contig.numel();
        torch::Tensor flat = contig.reshape({n});
        for (int64_t i = 0; i < n; ++i) {
          double v = 0.0; try { v = flat[i].item<double>(); } catch (...) { v = flat[i].item<float>(); }
          ofs << v;
          if (i + 1 < n) ofs << "\n";
        }
      };

      if (fo.lhsIsTensor) {
        Tensor t = readTensorFromFile(fo.file.text);
        env_.bind(fo.tensor, t);
        if (debug_) {
          std::ostringstream oss; oss << "Loaded tensor from '" << fo.file.text << "' into " << Environment::key(fo.tensor) << " shape=" << t.sizes();
          debugLog(oss.str());
        }
      } else {
        const auto &src = env_.lookup(fo.tensor);
        writeTensorToFile(fo.file.text, src);
        if (debug_) {
          std::ostringstream oss; oss << "Wrote tensor " << Environment::key(fo.tensor) << " shape=" << src.sizes() << " to '" << fo.file.text << "'";
            debugLog(oss.str());
          }
        }
      } else {
        // Unknown statement kind
        if (debug_) debugLog("Skipping non-tensor statement (not yet implemented)");
      }
    } // end for each preprocessed statement
  } // end for each original statement
}

// Resolve indices to concrete integer positions using either numeric indices
// or string labels (Uppercase identifiers). When createLabels=true, unseen
// labels are assigned new indices. When false, returns false if any label is unknown.
static bool resolveConcreteIndices(const TensorRef& ref,
                                   Environment& env,
                                   std::vector<int64_t>& idxs,
                                   bool createLabels) {
  idxs.clear();
  for (const auto& ix : ref.indices) {
    if (const auto* num = std::get_if<NumberLiteral>(&ix.value)) {
      try {
        long long v = std::stoll(num->text);
        if (v < 0) return false;
        idxs.push_back(static_cast<int64_t>(v));
      } catch (...) { return false; }
    } else if (const auto* id = std::get_if<Identifier>(&ix.value)) {
      if (createLabels) {
        int idx = env.internLabel(id->name);
        idxs.push_back(static_cast<int64_t>(idx));
      } else {
        int idx = 0;
        if (!env.getLabelIndex(id->name, idx)) return false;
        idxs.push_back(static_cast<int64_t>(idx));
      }
    } else {
      return false;
    }
  }
  return true;
}

void TensorLogicVM::execTensorEquation(const TensorEquation &eq) {
  // New refactored version using ExecutorRegistry
  try {
    std::string lhsName = Environment::key(eq.lhs);
    Tensor result = executor_registry_.execute(eq, env_, *torch_);

    // Special case: indexed LHS (e.g., avg[1] = expr)
    // Some executors (ScalarAssignExecutor) handle this internally and return the full tensor.
    // Others (ExpressionExecutor for non-literal RHS) just return the RHS value.
    // We need to detect the latter case and do the indexed assignment ourselves.
    if (!eq.lhs.indices.empty() && env_.has(lhsName)) {
      Tensor existingTensor = env_.lookup(lhsName);

      // Check if result is smaller than existing tensor (indicating executor returned RHS value)
      // In this case, we need to do indexed assignment
      if (result.numel() < existingTensor.numel() || result.dim() == 0) {
        // Build index list from LHS indices
        // Identifiers (free variables like 'i') become Slice()
        // NumberLiterals (concrete like '1') become concrete indices
        std::vector<torch::indexing::TensorIndex> indices;
        bool hasConcreteIndex = false;
        for (const auto& idx : eq.lhs.indices) {
          if (const auto* num = std::get_if<NumberLiteral>(&idx.value)) {
            long long v = std::stoll(num->text);
            indices.push_back(static_cast<int64_t>(v));
            hasConcreteIndex = true;
          } else if (std::holds_alternative<Identifier>(idx.value)) {
            // Free variable - use slice to cover all values in that dimension
            indices.push_back(torch::indexing::Slice());
          }
        }

        // Only do indexed assignment if we have at least one concrete index
        // Otherwise, it's a full tensor assignment
        if (hasConcreteIndex && !indices.empty()) {
          existingTensor.index_put_(indices, result);
          return;  // Don't rebind - we modified in-place
        }
      }
    }

    // Default case: bind result to environment
    env_.bind(eq.lhs, result);
  } catch (const ExecutionError& e) {
    if (debug_) {
      debugLog("Execution error: " + std::string(e.what()));
    }
    throw;
  }
}

void TensorLogicVM::execQuery(const Query &q) {
  using torch::indexing::TensorIndex;

  // Handle tensor queries
  if (std::holds_alternative<TensorRef>(q.target)) {
    const auto &ref = std::get<TensorRef>(q.target);
    const std::string name = Environment::key(ref);
    if (debug_) debugLog("Query: " + name);
    // Lookup (throws if missing)
    const auto &t = env_.lookup(ref);

    // If specific indices provided (numeric or label), print scalar value
    std::vector<int64_t> idxs;
    if (!ref.indices.empty() && resolveConcreteIndices(ref, env_, idxs, false)) {
      std::vector<TensorIndex> elemIdx;
      elemIdx.reserve(idxs.size());
      for (int64_t v : idxs) elemIdx.emplace_back(v);
      torch::Tensor elem = t.index(elemIdx);
      double val = 0.0;
      try { val = elem.item<double>(); } catch (...) { val = elem.item<float>(); }
      (*output_stream_) << name << "[";
      for (size_t i = 0; i < idxs.size(); ++i) { if (i) (*output_stream_) << ","; (*output_stream_) << idxs[i]; }
      (*output_stream_) << "] = " << val << std::endl;
      if (debug_) {
        std::ostringstream oss;
        oss << "Query tensor present: shape=" << t.sizes();
        debugLog(oss.str());
      }
      return;
    }

    // Otherwise, print entire tensor
    (*output_stream_) << name << " =\n" << t << std::endl;
    if (debug_) {
      std::ostringstream oss;
      oss << "Query tensor present: shape=" << t.sizes();
      debugLog(oss.str());
    }
    return;
  }

  // Handle Datalog queries - delegate to DatalogEngine
  datalog_engine_.query(q, *output_stream_);
}

} // namespace tl
