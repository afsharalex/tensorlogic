#include "TL/vm.hpp"
#include "TL/Runtime/Executors/ScalarAssignExecutor.hpp"
#include "TL/Runtime/Executors/ListLiteralExecutor.hpp"
#include "TL/Runtime/Executors/EinsumExecutor.hpp"
#include "TL/Runtime/Executors/IndexedProductExecutor.hpp"
#include "TL/Runtime/Executors/ReductionExecutor.hpp"
#include "TL/Runtime/Executors/PoolingExecutor.hpp"
#include "TL/Runtime/Executors/IdentityExecutor.hpp"
#include "TL/Runtime/Executors/ExpressionExecutor.hpp"

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
  : output_stream_(out), error_stream_(err) {
  torch_ = BackendFactory::create(BackendType::LibTorch);
  if (const char* env = std::getenv("TL_DEBUG")) {
    std::string v = env;
    for (auto &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "yes" || v == "on") {
      debug_ = true;
    }
  }
  initializeExecutors();
}

void TensorLogicVM::initializeExecutors() {
  // Register executors in order of priority (lower number = higher priority)
  executor_registry_.registerExecutor(std::make_unique<ScalarAssignExecutor>());       // 10
  executor_registry_.registerExecutor(std::make_unique<ListLiteralExecutor>());        // 20
  executor_registry_.registerExecutor(std::make_unique<EinsumExecutor>());             // 30
  executor_registry_.registerExecutor(std::make_unique<IndexedProductExecutor>());     // 35
  executor_registry_.registerExecutor(std::make_unique<ReductionExecutor>());          // 40
  executor_registry_.registerExecutor(std::make_unique<PoolingExecutor>());            // 50
  executor_registry_.registerExecutor(std::make_unique<IdentityExecutor>());           // 80
  executor_registry_.registerExecutor(std::make_unique<ExpressionExecutor>());         // 90
  // FallbackExecutor removed - no longer needed!

  // Pass debug settings to registry
  executor_registry_.setDebug(debug_);
  executor_registry_.setErrOut(error_stream_);
}

void TensorLogicVM::setDebug(bool enabled) { debug_ = enabled; }
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
    const BackendType be = router_.analyze(st);
    if (debug_) {
      debugLog(std::string("Router selected backend: ") + (be == BackendType::LibTorch ? "LibTorch" : "Unknown"));
    }

    if (std::holds_alternative<TensorEquation>(st)) {
      execTensorEquation(std::get<TensorEquation>(st));
    } else if (std::holds_alternative<Query>(st)) {
      // Ensure closure is up-to-date before answering queries
      saturateRules();
      execQuery(std::get<Query>(st));
    } else if (std::holds_alternative<DatalogFact>(st)) {
      const auto &f = std::get<DatalogFact>(st);
      const bool inserted = env_.addFact(f);
      if (inserted) closureDirty_ = true;
      if (debug_) {
        std::ostringstream oss;
        oss << "Added fact: " << f.relation.name << "(";
        for (size_t i = 0; i < f.constants.size(); ++i) {
          if (i) oss << ", ";
          oss << f.constants[i].text;
        }
        oss << ")";
        debugLog(oss.str());
      }
    } else if (std::holds_alternative<DatalogRule>(st)) {
      rules_.push_back(std::get<DatalogRule>(st));
      closureDirty_ = true;
      if (debug_) debugLog("Registered Datalog rule");
    } else if (std::holds_alternative<FileOperation>(st)) {
      const auto &fo = std::get<FileOperation>(st);
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
    Tensor result = executor_registry_.execute(eq, env_, *torch_);
    env_.bind(eq.lhs, result);
  } catch (const ExecutionError& e) {
    if (debug_) {
      debugLog("Execution error: " + std::string(e.what()));
    }
    throw;
  }
}

void TensorLogicVM::saturateRules() {
  if (!closureDirty_ || rules_.empty()) return;
  size_t totalNew = 0;
  size_t iter = 0;
  while (true) {
    size_t roundNew = 0;
    for (const auto &r : rules_) {
      roundNew += applyRule(r);
    }
    totalNew += roundNew;
    ++iter;
    if (roundNew == 0) break;
  }
  if (debug_) {
    std::ostringstream oss;
    oss << "Rule saturation finished after fixpoint.";
    debugLog(oss.str());
  }
  closureDirty_ = false;
}

// Helper: evaluate a simple expression into either numeric or string based on current Datalog bindings
static bool evalExprBinding(const ExprPtr &e,
                            const std::unordered_map<std::string, std::string> &binding,
                            std::string &outStr,
                            double &outNum,
                            bool &isNumeric) {
  if (!e) return false;
  const Expr &ex = *e;
  if (const auto *num = std::get_if<ExprNumber>(&ex.node)) {
    outStr = num->literal.text;
    try { outNum = std::stod(outStr); isNumeric = true; } catch (...) { isNumeric = false; }
    return true;
  }
  if (const auto *str = std::get_if<ExprString>(&ex.node)) {
    outStr = str->literal.text;
    try { outNum = std::stod(outStr); isNumeric = true; } catch (...) { isNumeric = false; }
    return true;
  }
  if (const auto *tr = std::get_if<ExprTensorRef>(&ex.node)) {
    // Treat lowercase scalar identifiers (no indices) as Datalog variables
    const std::string &name = tr->ref.name.name;
    const bool isVar = tr->ref.indices.empty() && !name.empty() && std::islower(static_cast<unsigned char>(name[0])) != 0;
    if (isVar) {
      auto it = binding.find(name);
      if (it == binding.end()) return false; // unbound
      outStr = it->second;
      try { outNum = std::stod(outStr); isNumeric = true; } catch (...) { isNumeric = false; }
      return true;
    }
    // Otherwise unsupported in condition
    return false;
  }
  if (const auto *paren = std::get_if<ExprParen>(&ex.node)) {
    return evalExprBinding(paren->inner, binding, outStr, outNum, isNumeric);
  }
  if (const auto *bin = std::get_if<ExprBinary>(&ex.node)) {
    // Minimal arithmetic support if both sides numeric
    std::string ls; double ln = 0; bool lnum = false;
    std::string rs; double rn = 0; bool rnum = false;
    if (!evalExprBinding(bin->lhs, binding, ls, ln, lnum)) return false;
    if (!evalExprBinding(bin->rhs, binding, rs, rn, rnum)) return false;
    if (!lnum || !rnum) return false;
    double res = 0;
    switch (bin->op) {
      case ExprBinary::Op::Add: res = ln + rn; break;
      case ExprBinary::Op::Sub: res = ln - rn; break;
      case ExprBinary::Op::Mul: res = ln * rn; break;
      case ExprBinary::Op::Div: if (rn == 0.0) return false; res = ln / rn; break;
    }
    outNum = res; isNumeric = true;
    std::ostringstream oss; oss << res; outStr = oss.str();
    return true;
  }
  // Lists and calls not supported in conditions
  return false;
}

static bool evalCondition(const DatalogCondition &cond,
                          const std::unordered_map<std::string, std::string> &binding) {
  std::string ls, rs; double ln = 0, rn = 0; bool lnum = false, rnum = false;
  if (!evalExprBinding(cond.lhs, binding, ls, ln, lnum)) return false;
  if (!evalExprBinding(cond.rhs, binding, rs, rn, rnum)) return false;

  auto doStrCmp = [&](const std::string &op) {
    if (op == "==") return ls == rs;
    if (op == "!=") return ls != rs;
    if (op == ">") return ls > rs;
    if (op == "<") return ls < rs;
    if (op == ">=") return ls >= rs;
    if (op == "<=") return ls <= rs;
    return false;
  };

  if ((cond.op == "==" || cond.op == "!=") && (!lnum || !rnum)) {
    return doStrCmp(cond.op);
  }
  if (lnum && rnum) {
    if (cond.op == "==") return ln == rn;
    if (cond.op == "!=") return ln != rn;
    if (cond.op == ">") return ln > rn;
    if (cond.op == "<") return ln < rn;
    if (cond.op == ">=") return ln >= rn;
    if (cond.op == "<=") return ln <= rn;
    return false;
  }
  // Mixed types with ordering: fallback to string compare
  return doStrCmp(cond.op);
}

size_t TensorLogicVM::applyRule(const DatalogRule &rule) {
  // Collect body atoms and conditions
  std::vector<DatalogAtom> bodyAtoms;
  std::vector<DatalogCondition> conditions;
  bodyAtoms.reserve(rule.body.size());
  for (const auto &el : rule.body) {
    if (const auto *a = std::get_if<DatalogAtom>(&el)) bodyAtoms.push_back(*a);
    else if (const auto *c = std::get_if<DatalogCondition>(&el)) conditions.push_back(*c);
  }
  if (bodyAtoms.empty()) return 0;

  size_t newCount = 0;

  // Depth-first join over body atoms
  std::unordered_map<std::string, std::string> binding;
  std::function<void(size_t)> dfs = [&](size_t idx) {
    if (idx == bodyAtoms.size()) {
      // Evaluate conditions as filters
      for (const auto &cond : conditions) {
        if (!evalCondition(cond, binding)) return; // reject this binding
      }
      // Build head tuple
      std::vector<std::string> headTuple;
      headTuple.reserve(rule.head.terms.size());
      for (const auto &t : rule.head.terms) {
        if (std::holds_alternative<StringLiteral>(t)) {
          headTuple.push_back(std::get<StringLiteral>(t).text);
        } else {
          const std::string &vn = std::get<Identifier>(t).name;
          auto it = binding.find(vn);
          if (it == binding.end()) {
            // Unsafe variable in head: skip
            return;
          }
          headTuple.push_back(it->second);
        }
      }
      if (env_.addFact(rule.head.relation.name, headTuple)) {
        ++newCount;
      }
      return;
    }

    const DatalogAtom &atom = bodyAtoms[idx];
    const auto &tuples = env_.facts(atom.relation.name);
    for (const auto &tup : tuples) {
      if (tup.size() != atom.terms.size()) continue;
      // Local modifications to binding; keep a list to rollback
      std::vector<std::string> assignedVars;
      bool ok = true;
      for (size_t i = 0; i < atom.terms.size(); ++i) {
        const auto &term = atom.terms[i];
        const std::string &val = tup[i];
        if (std::holds_alternative<StringLiteral>(term)) {
          if (std::get<StringLiteral>(term).text != val) { ok = false; break; }
        } else {
          const std::string &vn = std::get<Identifier>(term).name;
          auto it = binding.find(vn);
          if (it == binding.end()) {
            binding.emplace(vn, val);
            assignedVars.push_back(vn);
          } else if (it->second != val) {
            ok = false; break;
          }
        }
      }
      if (ok) {
        dfs(idx + 1);
      }
      // rollback
      for (const auto &vn : assignedVars) binding.erase(vn);
    }
  };

  dfs(0);
  return newCount;
}

void TensorLogicVM::execQuery(const Query &q) {
  using torch::indexing::TensorIndex;
  // Phase 1: support tensor ref queries and print results
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
  } else {
    const auto &atom = std::get<DatalogAtom>(q.target);

    // If this is a conjunctive Datalog query with optional comparisons, evaluate via join
    if (!q.body.empty()) {
      // Separate atoms and conditions; ensure first element is included already
      std::vector<DatalogAtom> atoms;
      std::vector<DatalogCondition> conditions;
      atoms.reserve(q.body.size());
      for (const auto &el : q.body) {
        if (const auto *a = std::get_if<DatalogAtom>(&el)) atoms.push_back(*a);
        else if (const auto *c = std::get_if<DatalogCondition>(&el)) conditions.push_back(*c);
      }
      if (atoms.empty()) {
        (*output_stream_) << "None" << std::endl;
        return;
      }

      // Determine variable output order across atoms by first appearance
      std::vector<std::string> varNames;
      std::unordered_set<std::string> seen;
      for (const auto &a : atoms) {
        for (const auto &t : a.terms) {
          if (const auto *id = std::get_if<Identifier>(&t)) {
            const std::string &vn = id->name;
            if (!vn.empty() && std::islower(static_cast<unsigned char>(vn[0])) != 0) {
              if (!seen.count(vn)) { seen.insert(vn); varNames.push_back(vn); }
            }
          }
        }
      }

      // DFS join similar to rules
      std::unordered_map<std::string, std::string> binding;
      bool anyPrinted = false;

      std::function<void(size_t)> dfs = [&](size_t idx) {
        if (idx == atoms.size()) {
          // Evaluate conditions
          for (const auto &cond : conditions) {
            if (!evalCondition(cond, binding)) return;
          }
          if (varNames.empty()) {
            (*output_stream_) << "True" << std::endl;
            anyPrinted = true;
            return;
          }
          if (varNames.size() == 1) {
            (*output_stream_) << binding[varNames[0]] << std::endl;
            anyPrinted = true;
          } else {
            for (size_t i = 0; i < varNames.size(); ++i) {
              if (i) (*output_stream_) << ", ";
              (*output_stream_) << binding[varNames[i]];
            }
            (*output_stream_) << std::endl;
            anyPrinted = true;
          }
          return;
        }
        const DatalogAtom &a = atoms[idx];
        const auto &tuples = env_.facts(a.relation.name);
        for (const auto &tup : tuples) {
          if (tup.size() != a.terms.size()) continue;
          std::vector<std::string> assigned;
          bool ok = true;
          for (size_t i = 0; i < a.terms.size(); ++i) {
            const auto &term = a.terms[i];
            const std::string &val = tup[i];
            if (std::holds_alternative<StringLiteral>(term)) {
              if (std::get<StringLiteral>(term).text != val) { ok = false; break; }
            } else {
              const std::string &vn = std::get<Identifier>(term).name;
              auto it = binding.find(vn);
              if (it == binding.end()) { binding.emplace(vn, val); assigned.push_back(vn); }
              else if (it->second != val) { ok = false; break; }
            }
          }
          if (ok) dfs(idx + 1);
          for (const auto &vn : assigned) binding.erase(vn);
        }
      };

      dfs(0);
      if (!anyPrinted) {
        // Ground conjunctive query with no satisfying assignment
        if (varNames.empty()) {
          (*output_stream_) << "False" << std::endl;
        } else {
          (*output_stream_) << "None" << std::endl;
        }
      }
      return;
    }

    const std::string rel = atom.relation.name;
    if (debug_) {
      std::ostringstream oss;
      oss << "Query over Datalog atom: " << rel << "(";
      for (size_t i = 0; i < atom.terms.size(); ++i) {
        if (i) oss << ", ";
        if (std::holds_alternative<Identifier>(atom.terms[i])) oss << std::get<Identifier>(atom.terms[i]).name;
        else oss << std::get<StringLiteral>(atom.terms[i]).text;
      }
      oss << ")?";
      debugLog(oss.str());
    }

    // Collect variable positions and names in order of first appearance
    std::vector<int> varPositions;
    std::vector<std::string> varNames;
    std::vector<std::optional<std::string>> constants(atom.terms.size());
    std::unordered_map<std::string, int> firstPos; // for repeated variable consistency

    for (size_t i = 0; i < atom.terms.size(); ++i) {
      if (std::holds_alternative<Identifier>(atom.terms[i])) {
        const std::string &vname = std::get<Identifier>(atom.terms[i]).name;
        if (!firstPos.count(vname)) {
          firstPos[vname] = static_cast<int>(i);
          varPositions.push_back(static_cast<int>(i));
          varNames.push_back(vname);
        }
      } else {
        constants[i] = std::get<StringLiteral>(atom.terms[i]).text;
      }
    }

    const auto &tuples = env_.facts(rel);
    auto matchesTuple = [&](const std::vector<std::string> &tuple) -> bool {
      if (tuple.size() != atom.terms.size()) return false;
      // Check constants
      for (size_t i = 0; i < constants.size(); ++i) {
        if (constants[i].has_value() && tuple[i] != *constants[i]) return false;
      }
      // Check repeated vars consistency
      std::unordered_map<std::string, std::string> bind;
      for (size_t i = 0; i < atom.terms.size(); ++i) {
        if (std::holds_alternative<Identifier>(atom.terms[i])) {
          const std::string &vn = std::get<Identifier>(atom.terms[i]).name;
          auto it = bind.find(vn);
          if (it == bind.end()) bind.emplace(vn, tuple[i]);
          else if (it->second != tuple[i]) return false;
        }
      }
      return true;
    };

    // Ground query (no variables): print True/False
    if (varNames.empty()) {
      bool any = false;
      for (const auto &tup : tuples) { if (matchesTuple(tup)) { any = true; break; } }
      (*output_stream_) << (any ? "True" : "False") << std::endl;
      return;
    }

    // Variable bindings: print each matching binding
    bool anyPrinted = false;
    for (const auto &tup : tuples) {
      if (!matchesTuple(tup)) continue;
      if (varNames.size() == 1) {
        (*output_stream_) << tup[varPositions[0]] << std::endl;
        anyPrinted = true;
      } else {
        // Print comma-separated values for the variables in first-appearance order
        for (size_t i = 0; i < varNames.size(); ++i) {
          if (i) (*output_stream_) << ", ";
          (*output_stream_) << tup[varPositions[i]];
        }
        (*output_stream_) << std::endl;
        anyPrinted = true;
      }
    }
    if (!anyPrinted) {
      (*output_stream_) << "None" << std::endl;
    }
  }
}

} // namespace tl
