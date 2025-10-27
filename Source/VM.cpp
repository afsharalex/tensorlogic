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
#include "TL/Runtime/ExecutorUtils.hpp"

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
  if (debug_) {
    debugLog("========== EXECUTE START ==========");
    debugLog("Total statements: " + std::to_string(program.statements.size()));
  }

  // BATCH PREPROCESSING: Collect virtual-indexed statements for batch processing
  std::vector<Statement> virtualIndexedStmts;
  std::vector<size_t> virtualIndexedIndices;
  std::vector<Statement> processedStatements;
  std::vector<size_t> processedIndices;

  // First pass: identify virtual-indexed statements
  for (size_t i = 0; i < program.statements.size(); ++i) {
    const auto &st = program.statements[i];

    bool isVirtualIndexed = false;
    if (std::holds_alternative<TensorEquation>(st)) {
      const auto& eq = std::get<TensorEquation>(st);

      // Check LHS for virtual indices
      for (const auto& ios : eq.lhs.indices) {
        // Check if it's an Index (not a Slice), then check if it's a VirtualIndex
        if (std::holds_alternative<Index>(ios.value)) {
          const auto& idx = std::get<Index>(ios.value);
          if (std::holds_alternative<VirtualIndex>(idx.value)) {
            isVirtualIndexed = true;
            break;
          }
        }
      }

      // IMPORTANT: RHS-only virtual indices should NOT be batched
      // They are handled by single-statement preprocessing which substitutes them with 0
      // Only batch equations that have virtual indices on BOTH LHS and RHS (recurrent equations)
      // Examples:
      //   - State[i, *t+1] = W[i,j] State[j, *t] + Input[i, t]  -> BATCH (LHS has *t)
      //   - Output = sigmoid(W_out[i] State[i, *5])  -> DON'T BATCH (only RHS has *5)

      // Note: We already checked LHS above and found no virtual indices
      // So this equation will go through normal preprocessing, not batch
    }

    if (isVirtualIndexed) {
      virtualIndexedStmts.push_back(st);
      virtualIndexedIndices.push_back(i);
    } else {
      processedStatements.push_back(st);
      processedIndices.push_back(i);
    }
  }

  // IMPORTANT: Execute non-virtual statements first so tensors are defined
  // This allows getIterationCount to find driving tensors like Input
  if (debug_) {
    debugLog("Executing " + std::to_string(processedStatements.size()) + " non-virtual statements first");
  }

  for (size_t i = 0; i < processedStatements.size(); ++i) {
    const auto &st = processedStatements[i];
    if (debug_) {
      debugLog("Non-virtual stmt " + std::to_string(i) + ": " + toString(st));
    }

    // FIRST: Preprocess statement (for non-virtual preprocessors)
    auto preprocessed = preprocessor_registry_.preprocess(st, env_);

    // THEN: Execute each preprocessed statement
    for (const auto &preprocessed_st : preprocessed) {
      const BackendType be = router_.analyze(preprocessed_st);
      if (debug_ && preprocessed.size() > 1) {
        debugLog("  Preprocessed: " + toString(preprocessed_st));
      }

      if (std::holds_alternative<TensorEquation>(preprocessed_st)) {
        execTensorEquation(std::get<TensorEquation>(preprocessed_st));
      } else if (std::holds_alternative<FixedPointLoop>(preprocessed_st)) {
        executeFixedPointLoop(std::get<FixedPointLoop>(preprocessed_st));
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
      } else if (std::holds_alternative<Query>(preprocessed_st)) {
        // Queries are processed in a separate loop after saturation (see line ~459)
        // Do nothing here - not an error
      } else {
        // Unknown statement kind
        if (debug_) debugLog("Warning: Unknown statement type, skipping");
      }
    } // end for each preprocessed statement
  } // end for each non-virtual statement

  // NOW batch preprocess and execute virtual-indexed statements
  // At this point, tensors like Input should be defined
  if (!virtualIndexedStmts.empty()) {
    if (debug_) {
      debugLog("Batch preprocessing " + std::to_string(virtualIndexedStmts.size()) + " virtual-indexed statements");
    }
    std::vector<Statement> expandedVirtual = VirtualIndexPreprocessor::preprocessBatch(virtualIndexedStmts, env_);

    if (debug_) {
      debugLog("Executing " + std::to_string(expandedVirtual.size()) + " expanded virtual statements");
    }

    for (size_t i = 0; i < expandedVirtual.size(); ++i) {
      const auto &st = expandedVirtual[i];

      if (std::holds_alternative<TensorEquation>(st)) {
        const auto& eq = std::get<TensorEquation>(st);
        if (debug_) {
          std::ostringstream oss;
          oss << "Virtual stmt " << i << ": " << Environment::key(eq.lhs) << " = ...";

          // Show LHS tensor shape if it exists
          std::string lhsName = Environment::key(eq.lhs);
          if (env_.has(lhsName)) {
            oss << " (existing shape: " << env_.lookup(lhsName).sizes() << ")";
          } else {
            oss << " (new tensor)";
          }
          debugLog(oss.str());
        }

        try {
          execTensorEquation(eq);
        } catch (const std::exception& e) {
          if (debug_) {
            debugLog("ERROR executing virtual stmt " + std::to_string(i));
            debugLog("  LHS: " + Environment::key(eq.lhs));
            debugLog("  Error: " + std::string(e.what()));
          }
          throw;
        }
      } else if (std::holds_alternative<FixedPointLoop>(st)) {
        const auto& loop = std::get<FixedPointLoop>(st);
        if (debug_) {
          debugLog("Virtual stmt " + std::to_string(i) + ": FixedPointLoop for " + loop.monitoredTensor);
        }
        try {
          executeFixedPointLoop(loop);
        } catch (const std::exception& e) {
          if (debug_) {
            debugLog("ERROR executing fixed-point loop " + std::to_string(i));
            debugLog("  Monitored tensor: " + loop.monitoredTensor);
            debugLog("  Error: " + std::string(e.what()));
          }
          throw;
        }
      } else if (std::holds_alternative<Query>(st)) {
        // Ensure closure is up-to-date before answering queries
        datalog_engine_.saturate();
        execQuery(std::get<Query>(st));
      }
    }
  }

  // Finally, execute any remaining queries that weren't virtual-indexed
  for (size_t i = 0; i < program.statements.size(); ++i) {
    const auto &st = program.statements[i];
    if (std::holds_alternative<Query>(st)) {
      // Ensure closure is up-to-date before answering queries
      datalog_engine_.saturate();
      execQuery(std::get<Query>(st));
    }
  }
}

// Resolve indices to concrete integer positions using either numeric indices
// or string labels (Uppercase identifiers). When createLabels=true, unseen
// labels are assigned new indices. When false, returns false if any label is unknown.
static bool resolveConcreteIndices(const TensorRef& ref,
                                   Environment& env,
                                   std::vector<int64_t>& idxs,
                                   bool createLabels) {
  idxs.clear();
  for (const auto& ios : ref.indices) {
    // Skip slices - only handle concrete Index values
    if (!std::holds_alternative<Index>(ios.value)) {
      return false;  // Has a slice, not fully concrete
    }
    const auto& ix = std::get<Index>(ios.value);

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
    if (debug_) {
      std::ostringstream oss;
      oss << "Executing: " << lhsName;
      if (!eq.lhs.indices.empty()) {
        oss << " with " << eq.lhs.indices.size() << " indices";
      }
      debugLog(oss.str());
    }
    Tensor result = executor_registry_.execute(eq, env_, *torch_);
    if (debug_) {
      std::ostringstream oss;
      oss << "  Result shape: " << result.sizes() << ", numel=" << result.numel();
      debugLog(oss.str());
    }

    // Special case: indexed LHS (e.g., avg[1] = expr, Input_proj2[i,t] = expr)
    // Some executors (ScalarAssignExecutor, ListLiteralExecutor) handle this internally and return the full tensor.
    // Others (ExpressionExecutor for non-literal RHS) just return the RHS value.
    // We need to detect the latter case and do the indexed assignment ourselves.
    if (!eq.lhs.indices.empty()) {
      // Check if all indices are concrete (number literals AND/OR uppercase labels)
      // ScalarAssignExecutor handles assignments where all indices are concrete/labels.
      // If there are any lowercase free variables, VM must handle indexed assignment.
      bool allConcreteOrLabelIndices = true;
      bool hasFreeVariables = false;

      for (const auto& ios : eq.lhs.indices) {
        // Slices are not concrete - require special handling
        if (!std::holds_alternative<Index>(ios.value)) {
          allConcreteOrLabelIndices = false;
          break;
        }
        const auto& idx = std::get<Index>(ios.value);

        if (std::holds_alternative<NumberLiteral>(idx.value)) {
          // Concrete numeric index - OK for ScalarAssignExecutor
          continue;
        } else if (auto* id = std::get_if<Identifier>(&idx.value)) {
          // Check if it's a label (uppercase) or a free variable (lowercase)
          if (!id->name.empty() && std::isupper(id->name[0])) {
            // Uppercase label - OK for ScalarAssignExecutor
            continue;
          } else {
            // Lowercase free variable - requires VM indexed assignment
            hasFreeVariables = true;
            break;
          }
        } else {
          // Other types (VirtualIndex, etc.) - not handled by ScalarAssignExecutor
          allConcreteOrLabelIndices = false;
          break;
        }
      }

      // If all indices are concrete/labels AND there are no free variables,
      // ScalarAssignExecutor or ListLiteralExecutor already handled the indexed assignment
      // and returned the full tensor. Just bind it.
      if (allConcreteOrLabelIndices && !hasFreeVariables) {
        if (debug_) {
          debugLog("  All concrete/label indices - executor handled assignment");
        }
        env_.bind(eq.lhs, result);
        return;
      }

      // Otherwise, we have free variables (identifiers) and need to do indexed assignment
      // Build index list from LHS indices
      // Identifiers (free variables like 'i') become Slice()
      // NumberLiterals (concrete like '1') become concrete indices
      std::vector<torch::indexing::TensorIndex> indices;
      std::vector<int64_t> concreteIndices;  // Track concrete index values
      std::vector<bool> isConcreteFlag;      // Track which positions are concrete
      bool hasConcreteIndex = false;

      for (const auto& ios : eq.lhs.indices) {
        // Handle both Index and Slice
        if (std::holds_alternative<Slice>(ios.value)) {
          // Convert TL slice to PyTorch slice with proper bounds
          const auto& tl_slice = std::get<Slice>(ios.value);
          indices.push_back(executor_utils::convertSlice(tl_slice));
          concreteIndices.push_back(-1);  // Placeholder (slices don't have a single concrete index)
          isConcreteFlag.push_back(false);  // Slices are not concrete single indices
        } else {
          const auto& idx = std::get<Index>(ios.value);
          if (const auto* num = std::get_if<NumberLiteral>(&idx.value)) {
            long long v = std::stoll(num->text);
            indices.push_back(static_cast<int64_t>(v));
            concreteIndices.push_back(static_cast<int64_t>(v));
            isConcreteFlag.push_back(true);
            hasConcreteIndex = true;
          } else if (std::holds_alternative<Identifier>(idx.value)) {
            // Free variable - use slice to cover all values in that dimension
            indices.push_back(torch::indexing::Slice());
            concreteIndices.push_back(-1);  // Placeholder
            isConcreteFlag.push_back(false);
          }
        }
      }

      if (!indices.empty()) {
        // Check if tensor exists and needs indexed assignment
        if (env_.has(lhsName)) {
          Tensor existingTensor = env_.lookup(lhsName);

          if (debug_) {
            std::ostringstream oss;
            oss << "  Indexed assignment (mixed): existing=" << existingTensor.sizes()
                << " result=" << result.sizes() << " hasConcreteIndex=" << hasConcreteIndex;
            debugLog(oss.str());
          }

          if (result.numel() < existingTensor.numel() || result.dim() == 0 || hasConcreteIndex) {
            // Check if we need to resize to accommodate the concrete indices
            std::vector<int64_t> requiredShape(existingTensor.sizes().begin(), existingTensor.sizes().end());
            bool needsResize = false;

            for (size_t i = 0; i < isConcreteFlag.size(); ++i) {
              if (isConcreteFlag[i]) {
                int64_t requiredSize = concreteIndices[i] + 1;
                if (i >= requiredShape.size()) {
                  requiredShape.resize(i + 1, 1);
                  requiredShape[i] = requiredSize;
                  needsResize = true;
                } else if (requiredShape[i] < requiredSize) {
                  requiredShape[i] = requiredSize;
                  needsResize = true;
                }
              }
            }

            // Resize if needed
            if (needsResize) {
              Tensor resizedTensor = torch::zeros(requiredShape, existingTensor.options());
              // Copy existing values
              if (existingTensor.numel() > 0) {
                std::vector<torch::indexing::TensorIndex> copyIndices;
                for (int i = 0; i < existingTensor.dim(); ++i) {
                  copyIndices.push_back(torch::indexing::Slice(0, existingTensor.size(i)));
                }
                resizedTensor.index_put_(copyIndices, existingTensor);
              }
              env_.bind(lhsName, resizedTensor);
              resizedTensor.index_put_(indices, result);
              return;
            } else {
              // No resize needed, just do indexed assignment
              existingTensor.index_put_(indices, result);
              return;
            }
          }
        } else if (hasConcreteIndex) {
          // Tensor doesn't exist yet, and we have concrete indices
          // We need to create it with appropriate size

          // Infer shape from indices and result
          std::vector<int64_t> shape;
          int resultDim = 0;
          for (size_t i = 0; i < isConcreteFlag.size(); ++i) {
            if (isConcreteFlag[i]) {
              shape.push_back(concreteIndices[i] + 1);
            } else {
              // Slice - use corresponding dimension from result
              if (resultDim < result.dim()) {
                shape.push_back(result.size(resultDim));
                resultDim++;
              } else {
                shape.push_back(1);
              }
            }
          }

          // Create new tensor and assign
          Tensor newTensor = torch::zeros(shape, result.options());
          newTensor.index_put_(indices, result);
          env_.bind(lhsName, newTensor);
          return;
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

// Recursive helper for expression substitution
void TensorLogicVM::substituteVirtualIndexInExpr(Expr &expr, int concreteTimeStep) {
  if (auto *ref = std::get_if<ExprTensorRef>(&expr.node)) {
    for (auto &ios : ref->ref.indices) {
      // Only substitute if it's an Index (not a Slice)
      if (std::holds_alternative<Index>(ios.value)) {
        auto &idx = std::get<Index>(ios.value);
        if (auto *virt = std::get_if<VirtualIndex>(&idx.value)) {
          int slot = concreteTimeStep + virt->offset;
          NumberLiteral num;
          num.text = std::to_string(slot);
          num.loc = idx.loc;
          idx.value = num;
        }
      }
    }
  } else if (auto *bin = std::get_if<ExprBinary>(&expr.node)) {
    substituteVirtualIndexInExpr(*bin->lhs, concreteTimeStep);
    substituteVirtualIndexInExpr(*bin->rhs, concreteTimeStep);
  } else if (auto *un = std::get_if<ExprUnary>(&expr.node)) {
    substituteVirtualIndexInExpr(*un->operand, concreteTimeStep);
  } else if (auto *call = std::get_if<ExprCall>(&expr.node)) {
    for (auto &arg : call->args) {
      substituteVirtualIndexInExpr(*arg, concreteTimeStep);
    }
  } else if (auto *paren = std::get_if<ExprParen>(&expr.node)) {
    substituteVirtualIndexInExpr(*paren->inner, concreteTimeStep);
  } else if (auto *list = std::get_if<ExprList>(&expr.node)) {
    for (auto &elem : list->elements) {
      substituteVirtualIndexInExpr(*elem, concreteTimeStep);
    }
  }
  // ExprNumber, ExprString: no substitution needed
}

// Helper to expand virtual indices for a single iteration
TensorEquation TensorLogicVM::substituteVirtualIndex(const TensorEquation &eq, int concreteTimeStep) {
  TensorEquation result = eq;

  // Substitute LHS virtual indices
  for (auto &ios : result.lhs.indices) {
    // Only substitute if it's an Index (not a Slice)
    if (std::holds_alternative<Index>(ios.value)) {
      auto &idx = std::get<Index>(ios.value);
      if (auto *virt = std::get_if<VirtualIndex>(&idx.value)) {
        int slot = concreteTimeStep + virt->offset;
        NumberLiteral num;
        num.text = std::to_string(slot);
        num.loc = idx.loc;
        idx.value = num;
      }
    }
  }

  // Substitute RHS virtual indices (in all expressions)
  for (auto &clause : result.clauses) {
    substituteVirtualIndexInExpr(*clause.expr, concreteTimeStep);
    if (clause.guard) {
      substituteVirtualIndexInExpr(**clause.guard, concreteTimeStep);
    }
  }

  return result;
}

void TensorLogicVM::executeFixedPointLoop(const FixedPointLoop &loop) {
  constexpr int ABSOLUTE_MAX = ABSOLUTE_MAX_ITERS;  // 10000
  constexpr int MAX_STABLE = MAX_CONSECUTIVE_STABLE;  // 10
  constexpr float TOLERANCE = CONVERGENCE_TOLERANCE;  // 0.0001f

  int consecutiveStableCount = 0;
  int totalIterations = 0;
  Tensor prevState;

  if (debug_) {
    std::ostringstream oss;
    oss << "Fixed-point loop for " << loop.monitoredTensor
        << " (tolerance=" << TOLERANCE
        << ", maxStable=" << MAX_STABLE << ")";
    debugLog(oss.str());
  }

  while (totalIterations < ABSOLUTE_MAX) {
    // Save previous state (after first iteration)
    if (totalIterations > 0 && env_.has(loop.monitoredTensor)) {
      prevState = env_.lookup(loop.monitoredTensor).clone();
    }

    // Execute one iteration by substituting virtual index with concrete timestep
    TensorEquation expandedEq = substituteVirtualIndex(loop.equation, totalIterations);
    execTensorEquation(expandedEq);

    totalIterations++;

    // Check convergence (after second iteration onwards)
    if (totalIterations > 1 && env_.has(loop.monitoredTensor)) {
      Tensor currentState = env_.lookup(loop.monitoredTensor);

      // Compute maximum absolute change across all elements
      float maxChange = (currentState - prevState).abs().max().item<float>();

      if (maxChange <= TOLERANCE) {
        // Value is stable - increment counter
        consecutiveStableCount++;

        if (consecutiveStableCount >= MAX_STABLE) {
          // Converged! Exit loop
          if (debug_) {
            std::ostringstream oss;
            oss << "  Converged after " << totalIterations
                << " iterations (change=" << maxChange << ")";
            debugLog(oss.str());
          }
          return;
        }
      } else {
        // Value changed significantly - reset stability counter
        consecutiveStableCount = 0;
      }
    }
  }

  // Hit absolute maximum without convergence
  if (debug_) {
    std::ostringstream oss;
    oss << "  Hit max iterations (" << ABSOLUTE_MAX
        << ") without convergence";
    debugLog(oss.str());
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
      // Special case: 0-dim (scalar) tensors cannot be indexed
      // If query is like avg[0]?, we treat [0] as a "no-op" index for scalars
      if (t.dim() == 0) {
        // Check that all indices are 0
        bool allZero = true;
        for (int64_t idx : idxs) {
          if (idx != 0) {
            allZero = false;
            break;
          }
        }
        if (!allZero) {
          throw std::runtime_error("Cannot index 0-dim tensor with non-zero indices: " + name);
        }
        // Just use the scalar value directly
        double val = 0.0;
        try { val = t.item<double>(); } catch (...) { val = t.item<float>(); }
        (*output_stream_) << name << "[";
        for (size_t i = 0; i < idxs.size(); ++i) { if (i) (*output_stream_) << ","; (*output_stream_) << idxs[i]; }
        (*output_stream_) << "] = " << val << std::endl;
        if (debug_) {
          std::ostringstream oss;
          oss << "Query tensor present: shape=" << t.sizes() << " (0-dim scalar)";
          debugLog(oss.str());
        }
        return;
      }

      // Normal case: index into multi-dimensional tensor
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
