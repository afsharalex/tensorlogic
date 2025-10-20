#include "TL/vm.hpp"

#include <stdexcept>
#include <torch/torch.h>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cctype>

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

// -------- BackendRouter --------

BackendType BackendRouter::analyze(const Statement &st) {
  // Phase 1: Tensor equations go to LibTorch. Others ignored for now.
  if (std::holds_alternative<TensorEquation>(st)) {
    return BackendType::LibTorch;
  }
  return BackendType::LibTorch; // default
}

// -------- TensorLogicVM --------

TensorLogicVM::TensorLogicVM() {
  torch_ = BackendFactory::create(BackendType::LibTorch);
  if (const char* env = std::getenv("TL_DEBUG")) {
    std::string v = env;
    for (auto &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "yes" || v == "on") {
      debug_ = true;
    }
  }
}

void TensorLogicVM::setDebug(bool enabled) { debug_ = enabled; }
bool TensorLogicVM::debug() const { return debug_; }
void TensorLogicVM::debugLog(const std::string &msg) const {
  if (debug_) {
    std::cout << "[VM] " << msg << std::endl;
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
      execQuery(std::get<Query>(st));
    } else {
      // File ops, datalog facts and rules are not handled in Phase 1 VM.
      // They can be added later; for now we just skip.
      if (debug_) debugLog("Skipping non-tensor statement (not yet implemented)");
    }
  }
}

static bool tryParseEinsumCall(const ExprPtr &rhs,
                               std::string &spec_out,
                               std::vector<Tensor> &inputs_out,
                               const Environment &env) {
  if (!rhs) return false;
  const Expr &e = *rhs;
  const auto *call = std::get_if<ExprCall>(&e.node);
  if (!call) return false;
  if (call->func.name != "einsum") return false;
  if (call->args.empty()) return false;

  // First arg must be a string literal (spec)
  const auto *specNode = std::get_if<ExprString>(&call->args[0]->node);
  if (!specNode) return false;
  spec_out = specNode->literal.text;

  // Remaining args are tensor refs that must exist in env
  inputs_out.clear();
  for (size_t i = 1; i < call->args.size(); ++i) {
    const auto *argRef = std::get_if<ExprTensorRef>(&call->args[i]->node);
    if (!argRef) return false;
    const std::string name = Environment::key(argRef->ref);
    if (!env.has(name)) {
      throw std::runtime_error("einsum uses unknown tensor: " + name);
    }
    inputs_out.push_back(env.lookup(name));
  }
  return true;
}

// --- Helpers for placeholder tensor materialization ---
static constexpr int64_t kDefaultExtent = 3;

static int64_t indexExtent(const Index &idx) {
  if (const auto *num = std::get_if<NumberLiteral>(&idx.value)) {
    try {
      return static_cast<int64_t>(std::stoll(num->text));
    } catch (...) {
      return kDefaultExtent;
    }
  }
  // Identifier or anything else → default extent
  return kDefaultExtent;
}

static std::vector<int64_t> shapeFromRef(const TensorRef &ref) {
  std::vector<int64_t> dims;
  dims.reserve(ref.indices.size());
  for (const auto &idx : ref.indices) {
    dims.push_back(indexExtent(idx));
  }
  return dims;
}

static Tensor placeholderForRef(const TensorRef &ref) {
  const auto dims = shapeFromRef(ref);
  if (dims.empty()) {
    return torch::tensor(0.0f);
  }
  return torch::randn(dims);
}

// Try to extract a TensorRef from an expression (unwrap parentheses)
static const ExprTensorRef* asExprTensorRef(const ExprPtr& ep) {
  if (!ep) return nullptr;
  const Expr* e = ep.get();
  // Unwrap parens
  const Expr* cur = e;
  while (true) {
    if (const auto* tr = std::get_if<ExprTensorRef>(&cur->node)) return tr;
    if (const auto* par = std::get_if<ExprParen>(&cur->node)) {
      if (!par->inner) return nullptr;
      cur = par->inner.get();
      continue;
    }
    return nullptr;
  }
}

static std::string labelsFromRef(const TensorRef& ref) {
  std::string s;
  s.reserve(ref.indices.size());
  for (const auto& idx : ref.indices) {
    if (const auto* id = std::get_if<Identifier>(&idx.value)) {
      if (!id->name.empty()) s.push_back(id->name[0]);
    } else if (const auto* num = std::get_if<NumberLiteral>(&idx.value)) {
      // Numeric indices do not have labels in einsum; use placeholder letters
      // This is a simplification for early examples (which use letter indices)
      (void)num; s.push_back('x');
    }
  }
  return s;
}

static bool tryLowerIndexedProductToEinsum(const TensorRef& lhs,
                                           const ExprPtr& rhs,
                                           std::string& spec_out,
                                           std::vector<Tensor>& inputs_out,
                                           Environment& env) {
  if (!rhs) return false;
  const Expr& e = *rhs;
  const auto* bin = std::get_if<ExprBinary>(&e.node);
  if (!bin || bin->op != ExprBinary::Op::Mul) return false;

  const auto* leftRef  = asExprTensorRef(bin->lhs);
  const auto* rightRef = asExprTensorRef(bin->rhs);
  if (!leftRef || !rightRef) return false;

  // Build einsum spec like "ij,jk->ik"
  const std::string a = labelsFromRef(leftRef->ref);
  const std::string b = labelsFromRef(rightRef->ref);
  const std::string out = labelsFromRef(lhs);
  if (a.empty() || b.empty() || out.empty()) return false;
  spec_out = a + "," + b + "->" + out;

  // Collect inputs, materializing placeholders as needed
  inputs_out.clear();
  const std::string leftName = Environment::key(leftRef->ref);
  if (!env.has(leftName)) env.bind(leftRef->ref, placeholderForRef(leftRef->ref));
  inputs_out.push_back(env.lookup(leftRef->ref));

  const std::string rightName = Environment::key(rightRef->ref);
  if (!env.has(rightName)) env.bind(rightRef->ref, placeholderForRef(rightRef->ref));
  inputs_out.push_back(env.lookup(rightRef->ref));

  return true;
}

// ---- Element-wise assignment helpers ----
static bool parseNumberLiteral(const ExprPtr& ep, double& out) {
  if (!ep) return false;
  const Expr* cur = ep.get();
  while (true) {
    if (const auto* num = std::get_if<ExprNumber>(&cur->node)) {
      try {
        out = std::stod(num->literal.text);
      } catch (...) { return false; }
      return true;
    }
    if (const auto* par = std::get_if<ExprParen>(&cur->node)) {
      if (!par->inner) return false;
      cur = par->inner.get();
      continue;
    }
    return false;
  }
}

static bool gatherNumericIndices(const TensorRef& ref, std::vector<int64_t>& idxs) {
  idxs.clear();
  for (const auto& idx : ref.indices) {
    if (const auto* num = std::get_if<NumberLiteral>(&idx.value)) {
      try {
        long long v = std::stoll(num->text);
        if (v < 0) return false;
        idxs.push_back(static_cast<int64_t>(v));
      } catch (...) { return false; }
    } else {
      return false; // not all numeric
    }
  }
  return true;
}

void TensorLogicVM::execTensorEquation(const TensorEquation &eq) {
  using torch::indexing::Slice;
  using torch::indexing::TensorIndex;
  const std::string lhsName = Environment::key(eq.lhs);
  if (debug_) debugLog("Execute TensorEquation: " + lhsName);

  // Case 1: Element-wise numeric assignment, e.g., W[0,1] = 2.0
  double rhsValue = 0.0;
  if (parseNumberLiteral(eq.rhs, rhsValue)) {
    std::vector<int64_t> idxs;
    if (eq.lhs.indices.empty()) {
      // Scalar bind
      Tensor t = torch::tensor(static_cast<float>(rhsValue));
      if (debug_) {
        std::ostringstream oss;
        oss << "Bind scalar " << lhsName << " = " << rhsValue;
        debugLog(oss.str());
      }
      env_.bind(eq.lhs, t);
      return;
    } else if (gatherNumericIndices(eq.lhs, idxs)) {
      // Determine required shape (max(index)+1 per dim)
      std::vector<int64_t> reqShape;
      reqShape.reserve(idxs.size());
      for (int64_t v : idxs) reqShape.push_back(v + 1);

      torch::Tensor t;
      const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
      if (!env_.has(lhsName)) {
        t = torch::zeros(reqShape, opts);
      } else {
        torch::Tensor cur = env_.lookup(lhsName);
        // Grow if needed
        bool needGrow = static_cast<size_t>(cur.dim()) != reqShape.size();
        if (!needGrow) {
          for (int64_t d = 0; d < cur.dim(); ++d) {
            if (cur.size(d) < reqShape[d]) { needGrow = true; break; }
          }
        }
        if (!needGrow) {
          t = cur.clone();
        } else {
          // New bigger tensor: use max(old, required) per dimension
          const size_t dims = std::max(static_cast<size_t>(cur.dim()), reqShape.size());
          std::vector<int64_t> newShape(dims, 1);
          for (size_t d = 0; d < dims; ++d) {
            const int64_t oldDim = (d < static_cast<size_t>(cur.dim())) ? cur.size(static_cast<int64_t>(d)) : 1;
            const int64_t reqDim = (d < reqShape.size()) ? reqShape[d] : 1;
            newShape[d] = std::max(oldDim, reqDim);
          }
          t = torch::zeros(newShape, opts);
          // Copy overlapping region from old tensor
          std::vector<TensorIndex> slices;
          slices.reserve(dims);
          for (size_t d = 0; d < dims; ++d) {
            const int64_t len = (d < static_cast<size_t>(cur.dim())) ? cur.size(static_cast<int64_t>(d)) : 1;
            slices.emplace_back(Slice(0, len));
          }
          if (!slices.empty()) {
            t.index_put_(slices, cur.index(slices));
          }
        }
      }
      // Write the element
      std::vector<TensorIndex> elemIdx;
      elemIdx.reserve(idxs.size());
      for (int64_t v : idxs) elemIdx.emplace_back(v);
      t.index_put_(elemIdx, torch::tensor(static_cast<float>(rhsValue), opts));

      if (debug_) {
        std::ostringstream oss;
        oss << "Set " << lhsName << "[";
        for (size_t i = 0; i < idxs.size(); ++i) { if (i) oss << ","; oss << idxs[i]; }
        oss << "] = " << rhsValue << ", shape now=" << t.sizes();
        debugLog(oss.str());
      }
      env_.bind(lhsName, t);
      return;
    }
    // If indices are not numeric, fallthrough to other handlers
  }

  // Support explicit einsum("spec", A, B, ...),
  // and implicit indexed products lowered to einsum.
  std::string spec;
  std::vector<Tensor> inputs;
  bool lowered = false;
  if (tryParseEinsumCall(eq.rhs, spec, inputs, env_)) {
    if (debug_) {
      std::ostringstream oss;
      oss << "Parsed explicit einsum: '" << spec << "' with inputs=";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &t = inputs[i];
        oss << (i ? "," : "") << t.sizes();
      }
      debugLog(oss.str());
    }
    lowered = true;
  } else if (tryLowerIndexedProductToEinsum(eq.lhs, eq.rhs, spec, inputs, env_)) {
    if (debug_) {
      std::ostringstream oss;
      oss << "Lowered indexed product to einsum: '" << spec << "' with inputs=";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &t = inputs[i];
        oss << (i ? "," : "") << t.sizes();
      }
      debugLog(oss.str());
    }
    lowered = true;
  }

  if (lowered) {
    const Tensor result = torch_->einsum(spec, inputs);
    if (debug_) {
      std::ostringstream oss;
      oss << "Result of einsum bound to " << lhsName << " with shape=" << result.sizes();
      debugLog(oss.str());
    }
    env_.bind(eq.lhs, result);
    return;
  }

  // If RHS is a direct tensor ref, treat as identity assignment.
  if (eq.rhs) {
    const Expr &e = *eq.rhs;
    if (const auto *eref = std::get_if<ExprTensorRef>(&e.node)) {
      const std::string srcName = Environment::key(eref->ref);
      if (!env_.has(srcName)) {
        // Materialize a placeholder tensor for unbound source
        const auto dims = shapeFromRef(eref->ref);
        if (debug_) {
          std::ostringstream oss;
          oss << "Materialize placeholder for " << srcName << " with shape=";
          oss << "[";
          for (size_t i = 0; i < dims.size(); ++i) {
            if (i) oss << "x";
            oss << dims[i];
          }
          oss << "]";
          debugLog(oss.str());
        }
        env_.bind(eref->ref, placeholderForRef(eref->ref));
      }
      const auto &src = env_.lookup(eref->ref);
      if (debug_) {
        std::ostringstream oss;
        oss << "Identity bind: " << lhsName << " = " << srcName << " shape=" << src.sizes();
        debugLog(oss.str());
      }
      env_.bind(eq.lhs, src);
      return;
    }
  }

  // Fallback: not supported yet, just ignore (Phase 1 skeleton)
  if (debug_) debugLog("RHS not supported yet; statement skipped");
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

    // If specific numeric indices provided, print scalar value
    std::vector<int64_t> idxs;
    if (!ref.indices.empty() && gatherNumericIndices(ref, idxs)) {
      std::vector<TensorIndex> elemIdx;
      elemIdx.reserve(idxs.size());
      for (int64_t v : idxs) elemIdx.emplace_back(v);
      torch::Tensor elem = t.index(elemIdx);
      double val = 0.0;
      try { val = elem.item<double>(); } catch (...) { val = elem.item<float>(); }
      std::cout << name << "[";
      for (size_t i = 0; i < idxs.size(); ++i) { if (i) std::cout << ","; std::cout << idxs[i]; }
      std::cout << "] = " << val << std::endl;
      if (debug_) {
        std::ostringstream oss;
        oss << "Query tensor present: shape=" << t.sizes();
        debugLog(oss.str());
      }
      return;
    }

    // Otherwise, print entire tensor
    std::cout << name << " =\n" << t << std::endl;
    if (debug_) {
      std::ostringstream oss;
      oss << "Query tensor present: shape=" << t.sizes();
      debugLog(oss.str());
    }
  } else {
    if (debug_) debugLog("Query over Datalog atom (not yet implemented)");
  }
}

} // namespace tl
