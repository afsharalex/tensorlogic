#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/VM.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <unordered_map>
#include <functional>

namespace tl {

namespace {

// Registry of built-in functions with their arity and implementation
struct FunctionInfo {
    int arity;
    std::function<torch::Tensor(const std::vector<torch::Tensor>&)> impl;
};

const std::unordered_map<std::string, FunctionInfo> BUILTIN_FUNCTIONS = {
    {"relu",    {1, [](const auto& args) { return torch::relu(args[0]); }}},
    {"sigmoid", {1, [](const auto& args) { return torch::sigmoid(args[0]); }}},
    {"tanh",    {1, [](const auto& args) { return torch::tanh(args[0]); }}},
    {"sqrt",    {1, [](const auto& args) { return torch::sqrt(args[0]); }}},
    {"abs",     {1, [](const auto& args) { return torch::abs(args[0]); }}},
    {"exp",     {1, [](const auto& args) { return torch::exp(args[0]); }}},
    {"log",     {1, [](const auto& args) { return torch::log(args[0]); }}},
    {"sin",     {1, [](const auto& args) { return torch::sin(args[0]); }}},
    {"cos",     {1, [](const auto& args) { return torch::cos(args[0]); }}},
    {"tan",     {1, [](const auto& args) { return torch::tan(args[0]); }}},
    {"asin",    {1, [](const auto& args) { return torch::asin(args[0]); }}},
    {"acos",    {1, [](const auto& args) { return torch::acos(args[0]); }}},
    {"atan",    {1, [](const auto& args) { return torch::atan(args[0]); }}},
    {"step",    {1, [](const auto& args) { return torch::gt(args[0], 0).to(torch::kFloat32); }}},
    {"softmax", {1, [](const auto& args) {
        const auto& x = args[0];
        if (x.dim() == 0) return torch::tensor(1.0f);
        return torch::softmax(x, std::max<int64_t>(0, x.dim() - 1));
    }}},
};

}  // anonymous namespace

    bool ExpressionExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=)
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        // Must have an RHS expression
        if (!eq.clauses[0].expr) return false;

        // This is a catch-all executor, so it can handle any expression
        // More specific executors should have run before this one
        return true;
    }

    Tensor ExpressionExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        // Evaluate the RHS expression
        Tensor val = evalExpr(eq.clauses[0].expr, eq.lhs, env, backend);

        // Element-wise assignment with label creation: only for simple numeric literals
        // Example: W[Alice] = 1.0, W[0] = 2.0
        std::vector<int64_t> idxs_assign;
        auto parsed_literal = executor_utils::tryParseNumericLiteral(eq.clauses[0].expr);

        if (!eq.lhs.indices.empty() && parsed_literal.has_value()) {
            // RHS is a simple numeric literal, so we can do element-wise assignment
            // Use resolveIndicesCreatingLabels to handle both numeric and label indices
            idxs_assign = executor_utils::resolveIndicesCreatingLabels(eq.lhs, env);

            if (!idxs_assign.empty()) {
                // Ensure destination tensor exists and is large enough
                const std::string& lhs_name = eq.lhs.name.name;
                Tensor t = executor_utils::ensureTensorSize(lhs_name, idxs_assign, env);

            // Ensure RHS is scalar; if not, reduce to scalar by sum
            if (val.dim() > 0) {
                if (val.numel() == 1) {
                    val = val.reshape({});
                } else {
                    val = torch::sum(val);
                }
            }

            // Write element
            const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
            std::vector<torch::indexing::TensorIndex> elemIdx;
            elemIdx.reserve(idxs_assign.size());
            for (int64_t v : idxs_assign) {
                elemIdx.emplace_back(v);
            }
                t.index_put_(elemIdx, val.to(opts.dtype()));

                return t;
            }
        }

        // If LHS is scalar but RHS evaluated to a tensor, auto-reduce by summing
        if (eq.lhs.indices.empty() && val.dim() > 0) {
            val = torch::sum(val);
        }

        return val;
    }

    Tensor ExpressionExecutor::evalExpr(const ExprPtr& ep, const TensorRef& lhsCtx,
                                       Environment& env, TensorBackend& backend) const {
        if (!ep) {
            throw ExecutionError("null expression");
        }

        const Expr& e = *ep;

        // Handle numeric literals
        if (const auto* num = std::get_if<ExprNumber>(&e.node)) {
            double v = 0.0;
            try {
                v = std::stod(num->literal.text);
            } catch (const std::invalid_argument& e) {
                throw ExecutionError("Invalid number format '" + num->literal.text + "': " + e.what());
            } catch (const std::out_of_range& e) {
                throw ExecutionError("Number out of range '" + num->literal.text + "': " + e.what());
            }
            return torch::tensor(static_cast<float>(v));
        }

        // Handle tensor references
        if (const auto* tr = std::get_if<ExprTensorRef>(&e.node)) {
            // Check if indices are bound variables (for guarded clause evaluation)
            bool hasBounden = false;
            for (const auto& ios : tr->ref.indices) {
                if (std::holds_alternative<tl::Slice>(ios.value)) {
                    // Slices are not bound variables
                    continue;
                }
                const auto& idx = std::get<Index>(ios.value);
                if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                    if (env.has(id->name)) {
                        hasBounden = true;
                        break;
                    }
                }
            }

            if (hasBounden) {
                // Resolve indices using bound values from environment
                Tensor base = env.lookup(tr->ref.name.name);
                if (tr->ref.indices.empty()) return base;

                std::vector<torch::indexing::TensorIndex> indices;
                for (const auto& ios : tr->ref.indices) {
                    if (std::holds_alternative<tl::Slice>(ios.value)) {
                        // TL slice - use full torch slice
                        indices.emplace_back(torch::indexing::Slice());
                    } else {
                        const auto& idx = std::get<Index>(ios.value);
                        if (const auto* num = std::get_if<NumberLiteral>(&idx.value)) {
                            long long v = 0;
                            try { v = std::stoll(num->text); } catch (...) { v = 0; }
                            indices.emplace_back(static_cast<int64_t>(v));
                        } else if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                            // Check if bound in environment
                            if (env.has(id->name)) {
                                Tensor idxTensor = env.lookup(id->name);
                                // Convert to integer index
                                int64_t idxValue = static_cast<int64_t>(idxTensor.item<float>());
                                indices.emplace_back(idxValue);
                            } else {
                                // Not bound, use as slice
                                indices.emplace_back(torch::indexing::Slice());
                            }
                        }
                    }
                }
                return base.index(indices);
            }

            // Normal case: use valueForRef
            return executor_utils::valueForRef(tr->ref, env);
        }

        // Handle parentheses
        if (const auto* par = std::get_if<ExprParen>(&e.node)) {
            return evalExpr(par->inner, lhsCtx, env, backend);
        }

        // Handle list literals (should be caught by ListLiteralExecutor, but handle here as fallback)
        if (const auto* lst = std::get_if<ExprList>(&e.node)) {
            // Build n-D tensor from nested list literal
            std::function<void(const ExprPtr&, std::vector<int64_t>&, std::vector<float>&)> collect =
                [&](const ExprPtr& ep2, std::vector<int64_t>& shape_out, std::vector<float>& flat_out) {
                const Expr& ex = *ep2;
                if (const auto* l = std::get_if<ExprList>(&ex.node)) {
                    const size_t n = l->elements.size();
                    std::vector<int64_t> child_shape;
                    bool first = true;
                    for (const auto& child : l->elements) {
                        std::vector<int64_t> cs;
                        collect(child, cs, flat_out);
                        if (first) {
                            child_shape = cs;
                            first = false;
                        } else if (child_shape != cs) {
                            throw ExecutionError("List literal is not rectangular (sub-shapes differ)");
                        }
                    }
                    shape_out.clear();
                    shape_out.push_back(static_cast<int64_t>(n));
                    shape_out.insert(shape_out.end(), child_shape.begin(), child_shape.end());
                    return;
                }
                // Leaf: evaluate numeric expression to scalar
                Tensor v = evalExpr(ep2, lhsCtx, env, backend);
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
                throw ExecutionError("List literal leaf must be a scalar expression");
            };

            std::vector<float> data;
            std::vector<int64_t> shape;
            std::vector<int64_t> top_shape;
            auto selfPtr = std::make_shared<Expr>(e);
            collect(selfPtr, top_shape, data);
            shape = top_shape;

            if (shape.empty()) {
                return torch::tensor(data.empty() ? 0.0f : data[0]);
            }
            return torch::tensor(data).reshape(shape);
        }

        // Handle function calls using registry
        if (const auto* call = std::get_if<ExprCall>(&e.node)) {
            auto it = BUILTIN_FUNCTIONS.find(call->func.name);
            if (it == BUILTIN_FUNCTIONS.end()) {
                throw ExecutionError("Unknown function: '" + call->func.name + "'");
            }

            const auto& funcInfo = it->second;
            if (call->args.size() != static_cast<size_t>(funcInfo.arity)) {
                throw ExecutionError(
                    call->func.name + "() expects " + std::to_string(funcInfo.arity) +
                    " argument(s), got " + std::to_string(call->args.size()));
            }

            // Evaluate arguments
            std::vector<Tensor> args;
            args.reserve(call->args.size());
            for (const auto& arg : call->args) {
                args.push_back(evalExpr(arg, lhsCtx, env, backend));
            }

            return funcInfo.impl(args);
        }

        // Handle binary operations
        if (const auto* bin = std::get_if<ExprBinary>(&e.node)) {
            using Op = ExprBinary::Op;

            // For multiplication, try to lower to einsum first
            // But skip if index variables are bound (for guarded clause evaluation)
            if (bin->op == Op::Mul) {
                // Check if any index variables in lhsCtx are bound
                bool hasBoundIndices = false;
                for (const auto& ios : lhsCtx.indices) {
                    if (std::holds_alternative<tl::Slice>(ios.value)) {
                        // Slices are not bound variables
                        continue;
                    }
                    const auto& idx = std::get<Index>(ios.value);
                    if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                        if (env.has(id->name)) {
                            hasBoundIndices = true;
                            break;
                        }
                    }
                }

                if (!hasBoundIndices) {
                    std::string spec;
                    std::vector<Tensor> inputs;
                    if (executor_utils::tryLowerIndexedProductToEinsum(lhsCtx, ep, spec, inputs, env)) {
                        return backend.einsum(spec, inputs);
                    }
                }
            }

            // Evaluate operands
            Tensor a = evalExpr(bin->lhs, lhsCtx, env, backend);
            Tensor b = evalExpr(bin->rhs, lhsCtx, env, backend);

            // Try einsum optimization for multiplication
            if (bin->op == Op::Mul) {
                if (auto result = tryEinsumOptimization(bin, lhsCtx, a, b, env, backend)) {
                    return *result;
                }
            }
            // Apply operation
            switch (bin->op) {
                case Op::Add:
                    return a + b;
                case Op::Sub:
                    return a - b;
                case Op::Div:
                    return a / b;
                case Op::Mod:
                    return torch::fmod(a, b);
                case Op::Mul:
                    return a * b;
                case Op::Pow:
                    return torch::pow(a, b);
                case Op::Lt:
                    return torch::lt(a, b).to(torch::kFloat32);
                case Op::Le:
                    return torch::le(a, b).to(torch::kFloat32);
                case Op::Gt:
                    return torch::gt(a, b).to(torch::kFloat32);
                case Op::Ge:
                    return torch::ge(a, b).to(torch::kFloat32);
                case Op::Eq:
                    return torch::eq(a, b).to(torch::kFloat32);
                case Op::Ne:
                    return torch::ne(a, b).to(torch::kFloat32);
                case Op::And:
                    // Logical AND: both must be non-zero
                    return torch::logical_and(torch::ne(a, 0), torch::ne(b, 0)).to(torch::kFloat32);
                case Op::Or:
                    // Logical OR: at least one must be non-zero
                    return torch::logical_or(torch::ne(a, 0), torch::ne(b, 0)).to(torch::kFloat32);
                default:
                    throw ExecutionError("Unknown binary operator");
            }
        }

        // Handle unary operations
        if (const auto* un = std::get_if<ExprUnary>(&e.node)) {
            using Op = ExprUnary::Op;
            Tensor operand = evalExpr(un->operand, lhsCtx, env, backend);

            switch (un->op) {
                case Op::Neg:
                    return -operand;
                case Op::Not:
                    // Logical NOT: zero if non-zero, one if zero
                    return torch::eq(operand, 0).to(torch::kFloat32);
                default:
                    throw ExecutionError("Unknown unary operator");
            }
        }

        throw ExecutionError("Unsupported expression node in ExpressionExecutor");
    }

    std::optional<Tensor> ExpressionExecutor::tryEinsumOptimization(
        const ExprBinary* bin,
        const TensorRef& lhsCtx,
        const Tensor& a,
        const Tensor& b,
        Environment& env,
        TensorBackend& backend) const {

        // Check if left operand is a simple tensor reference with free indices
        const auto* leftRef = executor_utils::asExprTensorRef(bin->lhs);
        if (!leftRef) {
            return std::nullopt;
        }

        // Collect free variable indices from left operand
        std::vector<std::string> leftIndices;
        for (const auto& ios : leftRef->ref.indices) {
            if (std::holds_alternative<Index>(ios.value)) {
                const auto& idx = std::get<Index>(ios.value);
                if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                    leftIndices.push_back(id->name);
                }
            }
        }

        // Collect free variable indices from LHS context
        std::vector<std::string> outIndices;
        for (const auto& ios : lhsCtx.indices) {
            if (std::holds_alternative<Index>(ios.value)) {
                const auto& idx = std::get<Index>(ios.value);
                if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                    outIndices.push_back(id->name);
                }
            }
        }

        // If we have indexed operations and dimensions match what we'd expect from einsum
        if (leftIndices.empty() || a.dim() != static_cast<int64_t>(leftIndices.size()) || b.dim() == 0) {
            return std::nullopt;
        }

        // Build index mapping
        static const std::string pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::unordered_map<std::string, char> labelMap;
        size_t nextChar = 0;

        // Map left indices
        std::string leftSpec;
        for (const auto& idx : leftIndices) {
            if (labelMap.find(idx) == labelMap.end()) {
                if (nextChar >= pool.size()) {
                    return std::nullopt; // Replaces goto fallback_multiply
                }
                labelMap[idx] = pool[nextChar++];
            }
            leftSpec += labelMap[idx];
        }

        // Map output indices
        std::string outSpec;
        for (const auto& idx : outIndices) {
            if (labelMap.find(idx) == labelMap.end()) {
                if (nextChar >= pool.size()) {
                    return std::nullopt; // Replaces goto fallback_multiply
                }
                labelMap[idx] = pool[nextChar++];
            }
            outSpec += labelMap[idx];
        }

        // Right tensor: figure out its indices
        // It should have dimensions for indices that appear in leftIndices but not in outIndices
        std::string rightSpec;
        for (const auto& idx : leftIndices) {
            // Only include indices not in output (these get contracted)
            if (std::find(outIndices.begin(), outIndices.end(), idx) == outIndices.end()) {
                rightSpec += labelMap[idx];
            }
        }

        // Check if dimensions match and validate einsum indices
        if (rightSpec.empty() || b.dim() != static_cast<int64_t>(rightSpec.size())) {
            return std::nullopt;
        }

        // IMPORTANT: Validate that all output indices appear in at least one input
        // Without this check, we could generate invalid einsum specs like "ab,b->c"
        // where 'c' doesn't appear in any input operand
        for (char c : outSpec) {
            if (leftSpec.find(c) == std::string::npos && rightSpec.find(c) == std::string::npos) {
                return std::nullopt;
            }
        }

        // Construct and execute einsum
        std::string spec = leftSpec + "," + rightSpec + "->" + outSpec;
        std::vector<Tensor> inputs = {a, b};
        return backend.einsum(spec, inputs);
    }

    std::string ExpressionExecutor::name() const {
        return "ExpressionExecutor";
    }

    int ExpressionExecutor::priority() const {
        return 90; // Low priority - catch-all for expressions not handled by specific executors
    }

} // namespace tl
