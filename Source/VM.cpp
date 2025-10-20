#include "TL/vm.hpp"

#include <stdexcept>
#include <torch/torch.h>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cctype>
#include <functional>

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
    } else {
      // File ops not yet implemented
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
      // Numeric indices should not be lowered into einsum labels in our VM.
      // We handle numeric indices via direct indexing instead.
      (void)num; s.push_back('x');
    }
  }
  return s;
}

static bool hasNumericIndices(const TensorRef& ref) {
  for (const auto& idx : ref.indices) {
    if (std::holds_alternative<NumberLiteral>(idx.value)) return true;
  }
  return false;
}

// Adapt a bound tensor view to match the number of indices in the reference by
// squeezing size-1 dimensions or unsqueezing leading dimensions as needed.
static Tensor adaptTensorToRef(const Tensor& t, const TensorRef& ref) {
  size_t want = ref.indices.size();
  int64_t have = t.dim();
  torch::Tensor r = t;
  if (have == static_cast<int64_t>(want)) return r;
  if (have > static_cast<int64_t>(want)) {
    // Remove all size-1 dims, then check
    r = r.squeeze();
    if (r.dim() == static_cast<int64_t>(want)) return r;
    // If still too many dims, we cannot safely drop non-singleton dims; keep as is
    return r;
  }
  // have < want: add leading singleton dims
  while (r.dim() < static_cast<int64_t>(want)) {
    r = r.unsqueeze(0);
  }
  return r;
}

// Return the value referenced by a TensorRef, applying numeric indices as
// integer indexing (which reduces rank), and leaving symbolic indices as slices.
static Tensor valueForRef(const TensorRef& ref, Environment& env) {
  using torch::indexing::Slice;
  using torch::indexing::TensorIndex;
  // Lookup base tensor
  torch::Tensor base = env.lookup(ref);
  // Ensure at least as many dims as indices by unsqueezing leading dims
  while (base.dim() < static_cast<int64_t>(ref.indices.size())) {
    base = base.unsqueeze(0);
  }
  if (ref.indices.empty()) return base;
  std::vector<TensorIndex> idx;
  idx.reserve(ref.indices.size());
  for (const auto& ind : ref.indices) {
    if (const auto* num = std::get_if<NumberLiteral>(&ind.value)) {
      long long v = 0;
      try { v = std::stoll(num->text); } catch (...) { v = 0; }
      idx.emplace_back(static_cast<int64_t>(v));
    } else {
      idx.emplace_back(Slice());
    }
  }
  return base.index(idx);
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

  // Do not lower if any numeric indices are present; handle via direct indexing instead
  if (hasNumericIndices(leftRef->ref) || hasNumericIndices(rightRef->ref) || hasNumericIndices(lhs)) {
    return false;
  }

  // Build einsum spec like "ij,jk->ik"
  const std::string a = labelsFromRef(leftRef->ref);
  const std::string b = labelsFromRef(rightRef->ref);
  const std::string out = labelsFromRef(lhs);
  if (a.empty() || b.empty()) return false;
  // Allow scalar outputs: empty 'out' means full contraction
  spec_out = a + "," + b + "->" + out;

  // Collect inputs, materializing placeholders as needed
  inputs_out.clear();
  const std::string leftName = Environment::key(leftRef->ref);
  if (!env.has(leftName)) env.bind(leftRef->ref, placeholderForRef(leftRef->ref));
  inputs_out.push_back(adaptTensorToRef(env.lookup(leftRef->ref), leftRef->ref));

  const std::string rightName = Environment::key(rightRef->ref);
  if (!env.has(rightName)) env.bind(rightRef->ref, placeholderForRef(rightRef->ref));
  inputs_out.push_back(adaptTensorToRef(env.lookup(rightRef->ref), rightRef->ref));

  return true;
}

// ---- Expression evaluation ----
static Tensor evalExpr(const ExprPtr& ep,
                       const TensorRef& lhsCtx,
                       Environment& env,
                       TensorBackend& backend) {
  if (!ep) throw std::runtime_error("null expression");
  const Expr& e = *ep;

  if (const auto* num = std::get_if<ExprNumber>(&e.node)) {
    double v = 0.0; try { v = std::stod(num->literal.text); } catch (...) {}
    return torch::tensor(static_cast<float>(v));
  }
  if (const auto* tr = std::get_if<ExprTensorRef>(&e.node)) {
    return valueForRef(tr->ref, env);
  }
  if (const auto* par = std::get_if<ExprParen>(&e.node)) {
    return evalExpr(par->inner, lhsCtx, env, backend);
  }
  if (const auto* lst = std::get_if<ExprList>(&e.node)) {
    // Build n-D tensor from nested list literal
    // Helper: recursively collect shape and data
    std::function<void(const ExprPtr&, std::vector<int64_t>&, std::vector<float>&)> collect = [&](const ExprPtr& ep,
                                                                                                   std::vector<int64_t>& shape_out,
                                                                                                   std::vector<float>& flat_out){
      const Expr& ex = *ep;
      if (const auto* l = std::get_if<ExprList>(&ex.node)) {
        const size_t n = l->elements.size();
        std::vector<int64_t> child_shape;
        bool first = true;
        for (const auto& child : l->elements) {
          std::vector<int64_t> cs;
          collect(child, cs, flat_out);
          if (first) { child_shape = cs; first = false; }
          else if (child_shape != cs) {
            throw std::runtime_error("List literal is not rectangular (sub-shapes differ)");
          }
        }
        shape_out.clear();
        shape_out.push_back(static_cast<int64_t>(n));
        shape_out.insert(shape_out.end(), child_shape.begin(), child_shape.end());
        return;
      }
      // Leaf: evaluate numeric expression to scalar
      Tensor v = evalExpr(ep, lhsCtx, env, backend);
      if (v.dim() == 0) {
        flat_out.push_back(v.item<float>());
        shape_out.clear();
        return;
      }
      if (v.numel() == 1) {
        flat_out.push_back(v.reshape({}).item<float>());
        shape_out.clear();
        return;
      }
      throw std::runtime_error("List literal leaf must be a scalar expression");
    };

    std::vector<float> data;
    std::vector<int64_t> shape;
    // Build from the current list expression
    std::vector<int64_t> top_shape;
    auto selfPtr = std::make_shared<Expr>(e);
    collect(selfPtr, top_shape, data);
    shape = top_shape;
    if (shape.empty()) {
      // Degenerate: single scalar list? Return scalar
      return torch::tensor(data.empty() ? 0.0f : data[0]);
    }
    return torch::tensor(data).reshape(shape);
  }
if (const auto* call = std::get_if<ExprCall>(&e.node)) {
    // Minimal function support
    auto need1 = [&](const char* fname){ if (call->args.size() != 1) throw std::runtime_error(std::string(fname) + "() expects 1 argument"); };
    if (call->func.name == "step") {
      need1("step");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      return torch::gt(x, 0).to(torch::kFloat32);
    } else if (call->func.name == "sqrt") {
      need1("sqrt");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      return torch::sqrt(x);
    } else if (call->func.name == "abs") {
      need1("abs");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      return torch::abs(x);
    } else if (call->func.name == "sigmoid") {
      need1("sigmoid");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      return torch::sigmoid(x);
    } else if (call->func.name == "tanh") {
      need1("tanh");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      return torch::tanh(x);
    } else if (call->func.name == "relu") {
      need1("relu");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      return torch::relu(x);
    } else if (call->func.name == "exp") {
      need1("exp");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      return torch::exp(x);
    } else if (call->func.name == "softmax") {
      need1("softmax");
      Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
      if (x.dim() == 0) return torch::tensor(1.0f);
      int64_t dim = std::max<int64_t>(0, x.dim() - 1);
      return torch::softmax(x, dim);
    }
    throw std::runtime_error("Unsupported function: " + call->func.name);
  }
  if (const auto* bin = std::get_if<ExprBinary>(&e.node)) {
    using Op = ExprBinary::Op;
    if (bin->op == Op::Mul) {
      // Prefer einsum lowering for indexed tensor refs
      std::string spec; std::vector<Tensor> inputs;
      ExprPtr mulExpr = std::make_shared<Expr>(e); // same node
      if (tryLowerIndexedProductToEinsum(lhsCtx, mulExpr, spec, inputs, env)) {
        return backend.einsum(spec, inputs);
      }
      Tensor a = evalExpr(bin->lhs, lhsCtx, env, backend);
      Tensor b = evalExpr(bin->rhs, lhsCtx, env, backend);
      return a * b;
    }
    Tensor a = evalExpr(bin->lhs, lhsCtx, env, backend);
    Tensor b = evalExpr(bin->rhs, lhsCtx, env, backend);
    switch (bin->op) {
      case Op::Add: return a + b;
      case Op::Sub: return a - b;
      case Op::Div: return a / b;
      case Op::Mul: default: return a * b;
    }
  }
  throw std::runtime_error("Unsupported expression node");
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

  // Case 2: List literal assignment to bare identifier, e.g., W = [ ... ] or nested lists
  if (eq.rhs && eq.lhs.indices.empty()) {
    const Expr &e = *eq.rhs;
    if (const auto *lst = std::get_if<ExprList>(&e.node)) {
      // Reuse nested list builder logic
      std::function<void(const ExprPtr&, std::vector<int64_t>&, std::vector<float>&)> collect = [&](const ExprPtr& ep,
                                                                                                    std::vector<int64_t>& shape_out,
                                                                                                    std::vector<float>& flat_out){
        const Expr& ex = *ep;
        if (const auto* l = std::get_if<ExprList>(&ex.node)) {
          const size_t n = l->elements.size();
          std::vector<int64_t> child_shape;
          bool first = true;
          for (const auto& child : l->elements) {
            std::vector<int64_t> cs;
            collect(child, cs, flat_out);
            if (first) { child_shape = cs; first = false; }
            else if (child_shape != cs) {
              throw std::runtime_error("List literal is not rectangular (sub-shapes differ)");
            }
          }
          shape_out.clear();
          shape_out.push_back(static_cast<int64_t>(n));
          shape_out.insert(shape_out.end(), child_shape.begin(), child_shape.end());
          return;
        }
        // Leaf: evaluate numeric expression to scalar
        Tensor v = evalExpr(ep, eq.lhs, env_, *torch_);
        if (v.dim() == 0) {
          flat_out.push_back(v.item<float>());
          shape_out.clear();
          return;
        }
        if (v.numel() == 1) {
          flat_out.push_back(v.reshape({}).item<float>());
          shape_out.clear();
          return;
        }
        throw std::runtime_error("List literal leaf must be a scalar expression");
      };

      std::vector<float> data;
      std::vector<int64_t> top_shape;
      // Build from 'e' by making a temporary shared_ptr to it
      auto selfPtr = std::make_shared<Expr>(e);
      collect(selfPtr, top_shape, data);
      Tensor t;
      if (top_shape.empty()) {
        t = torch::tensor(data.empty() ? 0.0f : data[0]);
      } else {
        t = torch::tensor(data).reshape(top_shape);
      }
      if (debug_) {
        std::ostringstream oss;
        oss << "Bind list literal to " << lhsName << " with shape=" << t.sizes();
        debugLog(oss.str());
      }
      env_.bind(eq.lhs, t);
      return;
    }
  }

  // Case 3: Reduction assignment like s = Y[i] (sum over i)
  if (eq.rhs) {
    const Expr &e = *eq.rhs;
    if (const auto *eref = std::get_if<ExprTensorRef>(&e.node)) {
      if (eq.lhs.indices.empty() && !eref->ref.indices.empty()) {
        const Tensor &src = env_.lookup(eref->ref);
        Tensor sumAll = torch::sum(src);
        if (debug_) {
          std::ostringstream oss;
          oss << "Reduce sum over all dims of " << Environment::key(eref->ref) << " → scalar bound to " << lhsName;
          debugLog(oss.str());
        }
        env_.bind(eq.lhs, sumAll);
        return;
      }
    }
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

  // General expression evaluation fallback (supports +,-,*,/, step, sqrt, etc.)
  if (eq.rhs) {
    try {
      Tensor val = evalExpr(eq.rhs, eq.lhs, env_, *torch_);

      // If LHS has fully numeric indices, treat as element-wise assignment using RHS value
      std::vector<int64_t> idxs_assign;
      if (!eq.lhs.indices.empty() && gatherNumericIndices(eq.lhs, idxs_assign)) {
        // Ensure destination tensor exists and is large enough
        std::vector<int64_t> reqShape;
        reqShape.reserve(idxs_assign.size());
        for (int64_t v : idxs_assign) reqShape.push_back(v + 1);
        torch::Tensor t;
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        if (!env_.has(lhsName)) {
          t = torch::zeros(reqShape, opts);
        } else {
          torch::Tensor cur = env_.lookup(lhsName);
          bool needGrow = static_cast<size_t>(cur.dim()) != reqShape.size();
          if (!needGrow) {
            for (int64_t d = 0; d < cur.dim(); ++d) {
              if (cur.size(d) < reqShape[d]) { needGrow = true; break; }
            }
          }
          if (!needGrow) {
            t = cur.clone();
          } else {
            const size_t dims = std::max(static_cast<size_t>(cur.dim()), reqShape.size());
            std::vector<int64_t> newShape(dims, 1);
            for (size_t d = 0; d < dims; ++d) {
              const int64_t oldDim = (d < static_cast<size_t>(cur.dim())) ? cur.size(static_cast<int64_t>(d)) : 1;
              const int64_t reqDim = (d < reqShape.size()) ? reqShape[d] : 1;
              newShape[d] = std::max(oldDim, reqDim);
            }
            t = torch::zeros(newShape, opts);
            // Copy overlapping region
            using torch::indexing::Slice; using torch::indexing::TensorIndex;
            std::vector<TensorIndex> slices; slices.reserve(dims);
            for (size_t d = 0; d < dims; ++d) {
              const int64_t len = (d < static_cast<size_t>(cur.dim())) ? cur.size(static_cast<int64_t>(d)) : 1;
              slices.emplace_back(Slice(0, len));
            }
            if (!slices.empty()) {
              t.index_put_(slices, cur.index(slices));
            }
          }
        }
        // Ensure RHS is scalar; if not, reduce to scalar by sum
        if (val.dim() > 0) {
          if (val.numel() == 1) val = val.reshape({});
          else val = torch::sum(val);
        }
        // Write element
        using torch::indexing::TensorIndex;
        std::vector<TensorIndex> elemIdx; elemIdx.reserve(idxs_assign.size());
        for (int64_t v : idxs_assign) elemIdx.emplace_back(v);
        t.index_put_(elemIdx, val.to(opts.dtype()))
         ;
        if (debug_) {
          std::ostringstream oss;
          oss << "Element-wise assign from expression: set " << lhsName << "[";
          for (size_t i = 0; i < idxs_assign.size(); ++i) { if (i) oss << ","; oss << idxs_assign[i]; }
          oss << "] = " << val.item<float>() << ", shape now=" << t.sizes();
          debugLog(oss.str());
        }
        env_.bind(lhsName, t);
        return;
      }

      // If LHS is scalar but RHS evaluated to a tensor, auto-reduce by summing all dims
      if (eq.lhs.indices.empty() && val.dim() > 0) {
        if (debug_) {
          std::ostringstream oss;
          oss << "Auto-reduce RHS to scalar via sum over all dims; original shape=" << val.sizes();
          debugLog(oss.str());
        }
        val = torch::sum(val);
      }
      if (debug_) {
        std::ostringstream oss;
        oss << "Evaluated expression bound to " << lhsName << " with shape=" << val.sizes();
        debugLog(oss.str());
      }
      env_.bind(eq.lhs, val);
      return;
    } catch (const std::exception&) {
      // fall through to other handlers
    }
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

size_t TensorLogicVM::applyRule(const DatalogRule &rule) {
  // Collect body atoms; ignore non-atom conditions for now (not used in 09_ examples)
  std::vector<DatalogAtom> bodyAtoms;
  bodyAtoms.reserve(rule.body.size());
  for (const auto &el : rule.body) {
    if (const auto *a = std::get_if<DatalogAtom>(&el)) bodyAtoms.push_back(*a);
  }
  if (bodyAtoms.empty()) return 0;

  size_t newCount = 0;

  // Depth-first join over body atoms
  std::unordered_map<std::string, std::string> binding;
  std::function<void(size_t)> dfs = [&](size_t idx) {
    if (idx == bodyAtoms.size()) {
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
    const auto &atom = std::get<DatalogAtom>(q.target);
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
      std::cout << (any ? "True" : "False") << std::endl;
      return;
    }

    // Variable bindings: print each matching binding
    bool anyPrinted = false;
    for (const auto &tup : tuples) {
      if (!matchesTuple(tup)) continue;
      if (varNames.size() == 1) {
        std::cout << tup[varPositions[0]] << std::endl;
        anyPrinted = true;
      } else {
        // Print comma-separated values for the variables in first-appearance order
        for (size_t i = 0; i < varNames.size(); ++i) {
          if (i) std::cout << ", ";
          std::cout << tup[varPositions[i]];
        }
        std::cout << std::endl;
        anyPrinted = true;
      }
    }
    if (!anyPrinted) {
      std::cout << "None" << std::endl;
    }
  }
}

} // namespace tl
