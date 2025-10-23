#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {
    namespace executor_utils {
        std::optional<double> tryParseNumericLiteral(const ExprPtr &expr) {
            if (!expr) return std::nullopt;

            const Expr *cur = expr.get();
            // Unwrap parentheses
            while (true) {
                if (const auto *num = std::get_if<ExprNumber>(&cur->node)) {
                    try {
                        return std::stod(num->literal.text);
                    } catch (...) {
                        return std::nullopt;
                    }
                }
                if (const auto *par = std::get_if<ExprParen>(&cur->node)) {
                    if (!par->inner) return std::nullopt;
                    cur = par->inner.get();
                    continue;
                }
                return std::nullopt;
            }
        }

        std::optional<std::vector<int64_t> > tryGatherNumericIndices(
            const TensorRef &ref, const Environment &env) {
            std::vector<int64_t> indices;

            for (const auto &idx: ref.indices) {
                if (const auto *num = std::get_if<NumberLiteral>(&idx.value)) {
                    try {
                        long long v = std::stoll(num->text);
                        if (v < 0) return std::nullopt;
                        indices.push_back(static_cast<int64_t>(v));
                    } catch (...) {
                        return std::nullopt;
                    }
                } else if (const auto *id = std::get_if<Identifier>(&idx.value)) {
                    // Try to resolve as label
                    int label_idx = 0;
                    if (!env.getLabelIndex(id->name, label_idx)) {
                        // Not all indices are numeric or resolvable
                        return std::nullopt;
                    }
                    indices.push_back(static_cast<int64_t>(label_idx));
                } else {
                    return std::nullopt;
                }
            }

            return indices;
        }

        std::vector<int64_t> resolveIndicesCreatingLabels(
            const TensorRef &ref, Environment &env) {
            std::vector<int64_t> indices;

            for (const auto &idx: ref.indices) {
                if (const auto *num = std::get_if<NumberLiteral>(&idx.value)) {
                    try {
                        long long v = std::stoll(num->text);
                        if (v < 0) return {};
                        indices.push_back(static_cast<int64_t>(v));
                    } catch (...) {
                        return {};
                    }
                } else if (const auto *id = std::get_if<Identifier>(&idx.value)) {
                    // Create or get label index
                    int label_idx = env.internLabel(id->name);
                    indices.push_back(static_cast<int64_t>(label_idx));
                } else {
                    return {};
                }
            }

            return indices;
        }


        Tensor ensureTensorSize(const std::string &name,
                                const std::vector<int64_t> &required_indices,
                                Environment &env) {
            // Calculate required shape (each index + 1)
            std::vector<int64_t> required_shape;
            for (int64_t idx: required_indices) {
                required_shape.push_back(idx + 1);
            }

            const auto opts = torch::TensorOptions().dtype(torch::kFloat32);

            // If tensor doesn't exist, create it
            if (!env.has(name)) {
                return torch::zeros(required_shape, opts);
            }

            // If tensor exists, check if it's large enough
            Tensor current = env.lookup(name);
            auto current_shape = current.sizes();

            bool needs_resize = false;
            std::vector<int64_t> new_shape = current_shape.vec();

            for (size_t i = 0; i < required_shape.size(); ++i) {
                if (i >= new_shape.size()) {
                    new_shape.push_back(required_shape[i]);
                    needs_resize = true;
                } else if (new_shape[i] < required_shape[i]) {
                    new_shape[i] = required_shape[i];
                    needs_resize = true;
                }
            }

            if (!needs_resize) {
                return current;
            }

            // Resize and copy
            Tensor resized = torch::zeros(new_shape, opts);

            // Calculate slice ranges for copying old data
            std::vector<torch::indexing::TensorIndex> slice_indices;
            for (int64_t dim_size: current_shape) {
                slice_indices.push_back(torch::indexing::Slice(0, dim_size));
            }

            resized.index_put_(slice_indices, current);

            return resized;
        }

        bool allConstants(const DatalogAtom &atom) {
            for (const auto &t: atom.terms) {
                if (std::holds_alternative<Identifier>(t)) {
                    const auto &id = std::get<Identifier>(t);
                    // Variable (lowercase) is not a constant
                    if (!id.name.empty() && std::islower(static_cast<unsigned char>(id.name[0]))) {
                        return false;
                    }
                }
                // StringLiteral is a constant, so skip
            }
            return true;
        }

        bool tryParseEinsumCall(const ExprPtr &expr,
                                std::string &spec_out,
                                std::vector<Tensor> &inputs_out,
                                const Environment &env) {
            if (!expr) return false;
            const Expr &e = *expr;
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
                const std::string name = argRef->ref.name.name;
                if (!env.has(name)) {
                    throw std::runtime_error("einsum uses unknown tensor: " + name);
                }
                inputs_out.push_back(env.lookup(name));
            }
            return true;
        }

        Tensor valueForRef(const TensorRef& ref, Environment& env) {
            using torch::indexing::Slice;
            using torch::indexing::TensorIndex;

            torch::Tensor base = env.lookup(ref);
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

        // Helper functions for tryLowerIndexedProductToEinsum
        static const ExprTensorRef* asExprTensorRef(const ExprPtr& ep) {
            if (!ep) return nullptr;
            const Expr* e = ep.get();
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

        static bool hasNumericIndices(const TensorRef& ref) {
            for (const auto& idx : ref.indices) {
                if (std::holds_alternative<NumberLiteral>(idx.value)) return true;
            }
            return false;
        }

        static constexpr int64_t kDefaultExtent = 3;
        static int64_t indexExtent(const Index &idx) {
            if (const auto *num = std::get_if<NumberLiteral>(&idx.value)) {
                try {
                    return static_cast<int64_t>(std::stoll(num->text));
                } catch (...) {
                    return kDefaultExtent;
                }
            }
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

        static Tensor adaptTensorToRef(const Tensor& t, const TensorRef& ref) {
            size_t want = ref.indices.size();
            int64_t have = t.dim();
            torch::Tensor r = t;
            if (have == static_cast<int64_t>(want)) return r;
            if (have > static_cast<int64_t>(want)) {
                r = r.squeeze();
                if (r.dim() == static_cast<int64_t>(want)) return r;
                return r;
            }
            while (r.dim() < static_cast<int64_t>(want)) {
                r = r.unsqueeze(0);
            }
            return r;
        }

        bool tryLowerIndexedProductToEinsum(const TensorRef& lhs,
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

            if (hasNumericIndices(leftRef->ref) || hasNumericIndices(rightRef->ref) || hasNumericIndices(lhs)) {
                return false;
            }

            auto collectNames = [](const TensorRef& r){
                std::vector<std::string> names; names.reserve(r.indices.size());
                for (const auto& idx : r.indices) {
                    if (const auto* id = std::get_if<Identifier>(&idx.value)) names.push_back(id->name);
                }
                return names;
            };
            const std::vector<std::string> leftNames = collectNames(leftRef->ref);
            const std::vector<std::string> rightNames = collectNames(rightRef->ref);
            const std::vector<std::string> outNames = collectNames(lhs);

            static const std::string pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            std::unordered_map<std::string, char> labelMap;
            size_t next = 0;
            auto mapSeq = [&](const std::vector<std::string>& seq){
                std::string s; s.reserve(seq.size());
                for (const auto& nm : seq) {
                    auto it = labelMap.find(nm);
                    if (it == labelMap.end()) {
                        if (next >= pool.size()) return std::string();
                        char c = pool[next++];
                        labelMap.emplace(nm, c);
                        s.push_back(c);
                    } else {
                        s.push_back(it->second);
                    }
                }
                return s;
            };

            const std::string a = mapSeq(leftNames);
            const std::string b = mapSeq(rightNames);
            const std::string out = mapSeq(outNames);
            if (a.empty() || b.empty()) return false;
            spec_out = a + "," + b + "->" + out;

            inputs_out.clear();
            const std::string leftName = leftRef->ref.name.name;
            if (!env.has(leftName)) env.bind(leftRef->ref, placeholderForRef(leftRef->ref));
            inputs_out.push_back(adaptTensorToRef(env.lookup(leftRef->ref), leftRef->ref));

            const std::string rightName = rightRef->ref.name.name;
            if (!env.has(rightName)) env.bind(rightRef->ref, placeholderForRef(rightRef->ref));
            inputs_out.push_back(adaptTensorToRef(env.lookup(rightRef->ref), rightRef->ref));

            return true;
        }
    } // namespace executor_utils
} // namespace tl
