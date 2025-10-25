#include "TL/Runtime/Preprocessors/VirtualIndexPreprocessor.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>
#include <set>
#include <map>
#include <stdexcept>

namespace tl {

namespace {

// Helper to check if an index list contains any virtual indices
bool hasVirtualIndices(const std::vector<Index>& indices) {
    for (const auto& idx : indices) {
        if (std::holds_alternative<VirtualIndex>(idx.value)) {
            return true;
        }
    }
    return false;
}

struct VirtualIndexCollector {
    // map name -> set of offsets used with that name
    std::map<std::string, std::set<int>> indicesByName;

    void visit(const Expr& expr) {
        std::visit([this](const auto& n) { (*this)(n); }, expr.node);
    }

    void operator()(const ExprTensorRef& ref) {
        for (const auto& idx: ref.ref.indices) {
            if (auto* vid = std::get_if<VirtualIndex>(&idx.value)) {
                indicesByName[vid->name.name].insert(vid->offset);
            }
        }
    }

    void operator()(const ExprNumber&) {}
    void operator()(const ExprString&) {}

    void operator()(const ExprList& lst) {
        for (const auto& e: lst.elements) visit(*e);
    }

    void operator()(const ExprParen& p) { visit(*p.inner); }

    void operator()(const ExprCall& c) {
        for (const auto& arg: c.args) visit(*arg);
    }

    void operator()(const ExprBinary& b) {
        visit(*b.lhs);
        visit(*b.rhs);
    }

    void operator()(const ExprUnary& u) {
        visit(*u.operand);
    }
};

    static std::map<std::string, std::set<int>> collectRhsVirtualIndices(const TensorEquation& eq) {
        VirtualIndexCollector c;
        for (const auto& cl : eq.clauses) {
            c.visit(*cl.expr);
            if (cl.guard) c.visit(**cl.guard);
        }
        return c.indicesByName;
    }

    static bool isAllDigits(const std::string& s) {
        return !s.empty() && std::all_of(s.begin(), s.end(), [](const unsigned char ch) { return std::isdigit(ch); });
    }

// Visitor to find all identifiers used as regular indices in an expression
struct IndexCollector {
    std::set<std::string> indices;

    void visit(const Expr& expr) {
        std::visit([this](const auto& node) { this->operator()(node); }, expr.node);
    }

    void operator()(const ExprTensorRef& ref) {
        for (const auto& idx : ref.ref.indices) {
            if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                indices.insert(id->name);
            }
        }
    }

    void operator()(const ExprNumber&) {}
    void operator()(const ExprString&) {}
    void operator()(const ExprList& lst) {
        for (const auto& e : lst.elements) visit(*e);
    }
    void operator()(const ExprParen& p) { visit(*p.inner); }
    void operator()(const ExprCall& c) {
        for (const auto& arg : c.args) visit(*arg);
    }
    void operator()(const ExprBinary& b) {
        visit(*b.lhs);
        visit(*b.rhs);
    }
    void operator()(const ExprUnary& u) {
        visit(*u.operand);
    }
};

// Visitor to check if an expression contains any virtual indices
struct VirtualIndexChecker {
    bool hasVirtual = false;

    void visit(const Expr& expr) {
        if (hasVirtual) return; // Short-circuit once found
        std::visit([this](const auto& node) { this->operator()(node); }, expr.node);
    }

    void operator()(const ExprTensorRef& ref) {
        if (hasVirtualIndices(ref.ref.indices)) {
            hasVirtual = true;
        }
    }

    void operator()(const ExprNumber&) {}
    void operator()(const ExprString&) {}
    void operator()(const ExprList& lst) {
        for (const auto& e : lst.elements) {
            if (hasVirtual) return;
            visit(*e);
        }
    }
    void operator()(const ExprParen& p) { visit(*p.inner); }
    void operator()(const ExprCall& c) {
        for (const auto& arg : c.args) {
            if (hasVirtual) return;
            visit(*arg);
        }
    }
    void operator()(const ExprBinary& b) {
        visit(*b.lhs);
        if (!hasVirtual) visit(*b.rhs);
    }
    void operator()(const ExprUnary& u) {
        visit(*u.operand);
    }
};

// Substitute indices in a TensorRef according to substitution map
TensorRef substituteIndices(const TensorRef& ref,
                           const std::map<std::string, int>& regularSubs,
                           const std::map<std::pair<std::string, int>, int>& virtualSubs) {
    TensorRef result = ref;
    for (auto& idx : result.indices) {
        if (auto* vid = std::get_if<VirtualIndex>(&idx.value)) {
            // Virtual index: look up in virtualSubs
            auto key = std::make_pair(vid->name.name, vid->offset);
            auto it = virtualSubs.find(key);
            if (it != virtualSubs.end()) {
                // Replace with concrete number
                NumberLiteral num;
                num.text = std::to_string(it->second);
                num.loc = idx.loc;
                idx.value = num;
            }
        } else if (auto* id = std::get_if<Identifier>(&idx.value)) {
            // Regular identifier: look up in regularSubs
            auto it = regularSubs.find(id->name);
            if (it != regularSubs.end()) {
                // Replace with concrete number
                NumberLiteral num;
                num.text = std::to_string(it->second);
                num.loc = idx.loc;
                idx.value = num;
            }
        }
    }
    return result;
}

// Substitute indices in an expression
ExprPtr substituteIndicesInExpr(const ExprPtr& expr,
                                const std::map<std::string, int>& regularSubs,
                                const std::map<std::pair<std::string, int>, int>& virtualSubs);

struct Substituter {
    const std::map<std::string, int>& regularSubs;
    const std::map<std::pair<std::string, int>, int>& virtualSubs;

    ExprPtr operator()(const ExprTensorRef& ref) {
        auto result = std::make_shared<Expr>();
        result->loc = ref.ref.loc;
        result->node = ExprTensorRef{substituteIndices(ref.ref, regularSubs, virtualSubs)};
        return result;
    }

    ExprPtr operator()(const ExprNumber& n) {
        auto result = std::make_shared<Expr>();
        result->loc = n.literal.loc;
        result->node = n;
        return result;
    }

    ExprPtr operator()(const ExprString& s) {
        auto result = std::make_shared<Expr>();
        result->loc = s.literal.loc;
        result->node = s;
        return result;
    }

    ExprPtr operator()(const ExprList& lst) {
        auto result = std::make_shared<Expr>();
        result->loc = lst.elements.empty() ? SourceLocation{} : lst.elements[0]->loc;
        ExprList newList;
        for (const auto& e : lst.elements) {
            newList.elements.push_back(visit(*e));
        }
        result->node = newList;
        return result;
    }

    ExprPtr operator()(const ExprParen& p) {
        auto result = std::make_shared<Expr>();
        result->loc = p.inner->loc;
        ExprParen newParen;
        newParen.inner = visit(*p.inner);
        result->node = newParen;
        return result;
    }

    ExprPtr operator()(const ExprCall& c) {
        auto result = std::make_shared<Expr>();
        result->loc = c.func.loc;
        ExprCall newCall;
        newCall.func = c.func;
        for (const auto& arg : c.args) {
            newCall.args.push_back(visit(*arg));
        }
        result->node = newCall;
        return result;
    }

    ExprPtr operator()(const ExprBinary& b) {
        auto result = std::make_shared<Expr>();
        result->loc = b.lhs->loc;
        ExprBinary newBin;
        newBin.op = b.op;
        newBin.lhs = visit(*b.lhs);
        newBin.rhs = visit(*b.rhs);
        result->node = newBin;
        return result;
    }

    ExprPtr operator()(const ExprUnary& u) {
        auto result = std::make_shared<Expr>();
        result->loc = u.operand->loc;
        ExprUnary newUnary;
        newUnary.op = u.op;
        newUnary.operand = visit(*u.operand);
        result->node = newUnary;
        return result;
    }

    ExprPtr visit(const Expr& expr) {
        return std::visit(*this, expr.node);
    }
};

ExprPtr substituteIndicesInExpr(const ExprPtr& expr,
                                const std::map<std::string, int>& regularSubs,
                                const std::map<std::pair<std::string, int>, int>& virtualSubs) {
    Substituter sub{regularSubs, virtualSubs};
    return sub.visit(*expr);
}

} // anonymous namespace

bool VirtualIndexPreprocessor::shouldPreprocess(const Statement& st, const Environment& env) const {
    // Only preprocess tensor equations with virtual indices
    if (!std::holds_alternative<TensorEquation>(st)) {
        return false;
    }
    const auto& eq = std::get<TensorEquation>(st);

    // Check for virtual indices in LHS
    if (hasVirtualIndices(eq.lhs.indices)) {
        return true;
    }

    const auto rhsV = collectRhsVirtualIndices(eq);
    return !rhsV.empty();

    // Check for virtual indices in RHS
    // for (const auto& clause : eq.clauses) {
    //     VirtualIndexChecker checker;
    //     checker.visit(*clause.expr);
    //     if (checker.hasVirtual) {
    //         return true;
    //     }
    //     if (clause.guard) {
    //         checker.visit(**clause.guard);
    //         if (checker.hasVirtual) {
    //             return true;
    //         }
    //     }
    // }
    //
    // return false;
}

std::vector<Statement> VirtualIndexPreprocessor::preprocess(const Statement& st, Environment& env) {
    if (!std::holds_alternative<TensorEquation>(st)) {
        return {st}; // Pass through non-equations
    }

    const auto& eq = std::get<TensorEquation>(st);

    // Step 1: Find virtual index info from LHS
    auto lhsVirtuals = findVirtualIndices(eq.lhs);
    if (lhsVirtuals.empty()) {
        // No virtual indices in LHS - check if there are any in RHS
        // If so, we need to substitute them with their concrete values (not expand)
        auto rhsV = collectRhsVirtualIndices(eq);
        if (rhsV.empty()) return {st};

        std::map<std::string, int> regularSubs; // empty
        std::map<std::pair<std::string, int>, int> virtualSubs;

        for (auto& [name, offsets] : rhsV) {
            for (int off : offsets) {
                // Only allow numeric *N (offset must be zero)
                if (isAllDigits(name) && off == 0) {
                    virtualSubs[{name, 0}] = std::stoi(name);
                } else {
                    throw std::runtime_error(
                        "RHS contains non-numeric virtual index '*" + name +
                        (off >= 0 ? "+" + std::to_string(off) : std::to_string(off)) +
                        "' without a driving LHS virtual index");
                }
            }
        }

        // bool hasRhsVirtuals = false;
        // for (const auto& clause : eq.clauses) {
        //     VirtualIndexChecker checker;
        //     checker.visit(*clause.expr);
        //     if (checker.hasVirtual) {
        //         hasRhsVirtuals = true;
        //         break;
        //     }
        // }
        //
        // if (hasRhsVirtuals) {
        //     // Substitute RHS virtual indices with their concrete values
        //     // Virtual index *N simply refers to the concrete value N
        //     std::map<std::string, int> regularSubs; // Empty - no regular index substitution
        //     std::map<std::pair<std::string, int>, int> virtualSubs;
        //
        //     // For virtual indices like *5, the parser creates VirtualIndex{name="5", offset=0}
        //     // For virtual indices like *t+1, the parser creates VirtualIndex{name="t", offset=1}
        //     // Build a mapping from all possible virtual index patterns to their concrete values
        //
        //     // Handle *N syntax (name=N as string, offset=0)
        //     for (int n = 0; n <= 20; ++n) {
        //         virtualSubs[{std::to_string(n), 0}] = n;
        //     }
        //
        //     // FIXME: We need to use the actual index names; not a list of common names and guess!
        //     // Handle *name+offset syntax (e.g., *t+1, *t-1)
        //     for (int offset = -10; offset <= 10; ++offset) {
        //         virtualSubs[{"t", offset}] = offset;
        //         // Add common names
        //         virtualSubs[{"i", offset}] = offset;
        //         virtualSubs[{"j", offset}] = offset;
        //         virtualSubs[{"k", offset}] = offset;
        //         virtualSubs[{"n", offset}] = offset;
        //     }
        //
        //     TensorEquation concreteEq = eq;
        //     for (size_t clauseIdx = 0; clauseIdx < concreteEq.clauses.size(); ++clauseIdx) {
        //         concreteEq.clauses[clauseIdx].expr =
        //             substituteIndicesInExpr(eq.clauses[clauseIdx].expr, regularSubs, virtualSubs);
        //         if (eq.clauses[clauseIdx].guard) {
        //             concreteEq.clauses[clauseIdx].guard =
        //                 substituteIndicesInExpr(*eq.clauses[clauseIdx].guard, regularSubs, virtualSubs);
        //         }
        //     }
        //
        //     return {concreteEq};
        // }

        // return {st}; // No virtual indices at all, pass through
    }

    // For now, we only support single virtual index on LHS
    if (lhsVirtuals.size() != 1) {
        throw std::runtime_error("Multiple virtual indices on LHS not yet supported");
    }

    std::string virtualIndexName = lhsVirtuals[0].first;
    int lhsOffset = lhsVirtuals[0].second;

    // Step 2: Find regular indices in RHS that match the virtual index name
    auto regularIndices = findRegularIndices(eq);

    // Check if the virtual index name appears as a regular index
    if (regularIndices.find(virtualIndexName) == regularIndices.end()) {
        throw std::runtime_error(
            "Virtual index '" + virtualIndexName + "' must appear as regular index in RHS to drive iteration");
    }

    // Step 3: Determine iteration count
    int iterationCount = getIterationCount(virtualIndexName, env, eq);

    // Step 4: Pre-allocate storage for LHS tensor
    preallocateStorage(eq, env, virtualIndexName, lhsOffset, iterationCount);

    // Step 5: Expand into concrete statements
    std::vector<Statement> result;

    auto rhsV = collectRhsVirtualIndices(eq);

    for (int i = 0; i < iterationCount; ++i) {
        std::map<std::string, int> regularSubs {{ virtualIndexName, i }};
        std::map<std::pair<std::string, int>, int> virtualSubs;

        // Only create substitutions for the offsets that are actually used
        if (auto it = rhsV.find(virtualIndexName); it != rhsV.end()) {
            for (int off : it->second) {
                virtualSubs[{virtualIndexName, off}] = i + off;
            }
        }

        // Optional: also substitute numeric *N on RHS if present
        for (auto& [name, offs] : rhsV) {
            if (name == virtualIndexName) continue;
            for (int off : offs) {
                if (isAllDigits(name) && off == 0) {
                    virtualSubs[{name, 0}] = std::stoi(name);
                } else {
                    // If other virtual-index names appear on RHS, decide policy:
                    // either forbid or support if their base also appears as a regular index with a known value.
                    // For now, forbid to keep semantics clear:
                    throw std::runtime_error(
                        "RHS contains unrelated virtual index '*" + name +
                        (off >= 0 ? "+" + std::to_string(off) : std::to_string(off)) + "'");
                }
            }
        }

        TensorEquation concreteEq = eq;
        concreteEq.lhs = substituteIndices(eq.lhs, regularSubs, virtualSubs);
        for (size_t clauseIdx = 0; clauseIdx < concreteEq.clauses.size(); ++clauseIdx) {
            concreteEq.clauses[clauseIdx].expr = substituteIndicesInExpr(eq.clauses[clauseIdx].expr, regularSubs, virtualSubs);
            if (eq.clauses[clauseIdx].guard) {
                concreteEq.clauses[clauseIdx].guard = substituteIndicesInExpr(*eq.clauses[clauseIdx].guard, regularSubs, virtualSubs);
            }
        }
        result.push_back(concreteEq);
        // // Create substitution maps
        // std::map<std::string, int> regularSubs;
        // regularSubs[virtualIndexName] = i;
        //
        // std::map<std::pair<std::string, int>, int> virtualSubs;
        // // Add mappings for common offsets
        // for (int offset = -10; offset <= 10; ++offset) {
        //     virtualSubs[{virtualIndexName, offset}] = i + offset;
        // }
        //
        // // Substitute indices in equation
        // TensorEquation concreteEq = eq;
        // concreteEq.lhs = substituteIndices(eq.lhs, regularSubs, virtualSubs);
        //
        // for (size_t clauseIdx = 0; clauseIdx < concreteEq.clauses.size(); ++clauseIdx) {
        //     concreteEq.clauses[clauseIdx].expr =
        //         substituteIndicesInExpr(eq.clauses[clauseIdx].expr, regularSubs, virtualSubs);
        //     if (eq.clauses[clauseIdx].guard) {
        //         concreteEq.clauses[clauseIdx].guard =
        //             substituteIndicesInExpr(*eq.clauses[clauseIdx].guard, regularSubs, virtualSubs);
        //     }
        // }
        //
        // result.push_back(concreteEq);
    }

    return result;
}

bool VirtualIndexPreprocessor::isVirtualIndex(const Index& idx) {
    return std::holds_alternative<VirtualIndex>(idx.value);
}

std::optional<std::pair<std::string, int>> VirtualIndexPreprocessor::getVirtualIndexInfo(const Index& idx) {
    if (!std::holds_alternative<VirtualIndex>(idx.value)) {
        return std::nullopt;
    }
    const auto& vidx = std::get<VirtualIndex>(idx.value);
    return std::make_pair(vidx.name.name, vidx.offset);
}

std::vector<std::pair<std::string, int>> VirtualIndexPreprocessor::findVirtualIndices(const TensorRef& ref) {
    std::vector<std::pair<std::string, int>> result;
    for (size_t i = 0; i < ref.indices.size(); ++i) {
        if (auto info = getVirtualIndexInfo(ref.indices[i])) {
            result.push_back({info->first, info->second});
        }
    }
    return result;
}

std::set<std::string> VirtualIndexPreprocessor::findRegularIndices(const TensorEquation& eq) {
    IndexCollector collector;
    for (const auto& clause : eq.clauses) {
        collector.visit(*clause.expr);
        if (clause.guard) {
            collector.visit(**clause.guard);
        }
    }
    return collector.indices;
}

int VirtualIndexPreprocessor::getIterationCount(const std::string& indexName, const Environment& env,
                                                 const TensorEquation& eq) {
    // Find a tensor indexed by the driving index and determine its size
    for (const auto& clause : eq.clauses) {
        // Try to find a tensor ref with this index
        struct TensorRefFinder {
            const std::string& targetIndex;
            std::optional<TensorRef> found;

            void visit(const Expr& expr) {
                std::visit([this](const auto& node) { this->operator()(node); }, expr.node);
            }

            void operator()(const ExprTensorRef& ref) {
                for (const auto& idx : ref.ref.indices) {
                    if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                        if (id->name == targetIndex) {
                            found = ref.ref;
                            return;
                        }
                    }
                }
            }

            void operator()(const ExprNumber&) {}
            void operator()(const ExprString&) {}
            void operator()(const ExprList& lst) {
                for (const auto& e : lst.elements) visit(*e);
            }
            void operator()(const ExprParen& p) { visit(*p.inner); }
            void operator()(const ExprCall& c) {
                for (const auto& arg : c.args) visit(*arg);
            }
            void operator()(const ExprBinary& b) {
                visit(*b.lhs);
                if (!found) visit(*b.rhs);
            }
            void operator()(const ExprUnary& u) { visit(*u.operand); }
        } finder{indexName, std::nullopt};

        finder.visit(*clause.expr);

        if (finder.found) {
            const TensorRef& ref = *finder.found;

            // Look up the tensor
            if (!env.has(ref)) {
                throw std::runtime_error(
                    "Cannot determine iteration count: tensor '" +
                    Environment::key(ref) + "' not found in environment");
            }

            const Tensor& t = env.lookup(ref);

            // Find which dimension corresponds to the driving index
            int dimIdx = -1;
            for (size_t i = 0; i < ref.indices.size(); ++i) {
                if (const auto* id = std::get_if<Identifier>(&ref.indices[i].value)) {
                    if (id->name == indexName) {
                        dimIdx = static_cast<int>(i);
                        break;
                    }
                }
            }

            if (dimIdx >= 0 && dimIdx < t.dim()) {
                return static_cast<int>(t.size(dimIdx));
            }
        }
    }

    // Fallback: return 10 as default
    return 10;
}

void VirtualIndexPreprocessor::preallocateStorage(const TensorEquation& eq, Environment& env,
                                                   const std::string& virtualIndexName,
                                                   int lhsOffset, int iterationCount) {
    std::string lhsTensorName = Environment::key(eq.lhs);

    // Determine the size needed for the virtual dimension
    int virtualDimSize = iterationCount + std::abs(lhsOffset);

    // Find which dimension has the virtual index
    int virtualDim = -1;
    for (size_t i = 0; i < eq.lhs.indices.size(); ++i) {
        if (std::holds_alternative<VirtualIndex>(eq.lhs.indices[i].value)) {
            virtualDim = static_cast<int>(i);
            break;
        }
    }

    if (virtualDim == -1) {
        throw std::runtime_error("No virtual index found in LHS during preallocation");
    }

    if (!env.has(lhsTensorName)) {
        // Tensor doesn't exist yet - can't preallocate without knowing other dimensions
        // Let normal indexed assignment handle initial creation
        return;
    }

    // Tensor exists - resize if the virtual dimension needs expansion
    Tensor existingTensor = env.lookup(lhsTensorName);

    // Check if the virtual dimension needs to be expanded
    if (virtualDim >= existingTensor.dim() || existingTensor.size(virtualDim) < virtualDimSize) {
        // Build new shape
        std::vector<int64_t> newShape;
        for (int d = 0; d < existingTensor.dim(); ++d) {
            if (d == virtualDim) {
                newShape.push_back(virtualDimSize);
            } else {
                newShape.push_back(existingTensor.size(d));
            }
        }

        // If virtualDim is beyond current dimensions, we need to add dimensions
        while (static_cast<int>(newShape.size()) <= virtualDim) {
            newShape.push_back(1);
        }
        newShape[virtualDim] = virtualDimSize;

        // Create new tensor and copy old data
        Tensor newTensor = torch::zeros(newShape, existingTensor.options());

        // Build slice indices to copy old data
        std::vector<torch::indexing::TensorIndex> sliceIndices;
        for (int d = 0; d < existingTensor.dim(); ++d) {
            sliceIndices.push_back(torch::indexing::Slice(0, existingTensor.size(d)));
        }

        newTensor.index(sliceIndices).copy_(existingTensor);

        env.bind(lhsTensorName, newTensor);
    }
}

} // namespace tl
