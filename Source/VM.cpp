#include "TL/vm.hpp"

#include <stdexcept>
#include <torch/torch.h>

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
}

void TensorLogicVM::execute(const Program &program) {
  for (const auto &st : program.statements) {
    if (std::holds_alternative<TensorEquation>(st)) {
      execTensorEquation(std::get<TensorEquation>(st));
    } else if (std::holds_alternative<Query>(st)) {
      execQuery(std::get<Query>(st));
    } else {
      // File ops, datalog facts and rules are not handled in Phase 1 VM.
      // They can be added later; for now we just skip.
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

void TensorLogicVM::execTensorEquation(const TensorEquation &eq) {
  // Support explicit einsum("spec", A, B, ...),
  // and implicit indexed products lowered to einsum.
  std::string spec;
  std::vector<Tensor> inputs;
  if (tryParseEinsumCall(eq.rhs, spec, inputs, env_) ||
      tryLowerIndexedProductToEinsum(eq.lhs, eq.rhs, spec, inputs, env_)) {
    const Tensor result = torch_->einsum(spec, inputs);
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
        env_.bind(eref->ref, placeholderForRef(eref->ref));
      }
      const auto &src = env_.lookup(eref->ref);
      env_.bind(eq.lhs, src);
      return;
    }
  }

  // Fallback: not supported yet, just ignore (Phase 1 skeleton)
}

void TensorLogicVM::execQuery(const Query &q) {
  // Phase 1: only support tensor ref query by ensuring it's materialized.
  if (std::holds_alternative<TensorRef>(q.target)) {
    const auto &ref = std::get<TensorRef>(q.target);
    // Lookup to trigger error if missing; ignore otherwise.
    (void)env_.lookup(ref);
  }
}

} // namespace tl
