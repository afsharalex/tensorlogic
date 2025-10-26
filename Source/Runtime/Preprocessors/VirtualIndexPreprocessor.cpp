#include "TL/Runtime/Preprocessors/VirtualIndexPreprocessor.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>
#include <set>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <sstream>

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
    // map (tensorName, virtualIndexName) -> set of offsets
    std::map<std::pair<std::string, std::string>, std::set<int>> tensorVirtualIndices;

    void visit(const Expr& expr) {
        std::visit([this](const auto& n) { (*this)(n); }, expr.node);
    }

    void operator()(const ExprTensorRef& ref) {
        std::string tensorName = ref.ref.name.name;
        for (const auto& idx: ref.ref.indices) {
            if (auto* vid = std::get_if<VirtualIndex>(&idx.value)) {
                tensorVirtualIndices[{tensorName, vid->name.name}].insert(vid->offset);
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

static std::map<std::pair<std::string, std::string>, std::set<int>> collectRhsVirtualIndices(const TensorEquation& eq) {
    VirtualIndexCollector c;
    for (const auto& cl : eq.clauses) {
        c.visit(*cl.expr);
        if (cl.guard) c.visit(**cl.guard);
    }
    return c.tensorVirtualIndices;
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

// Structure to track virtual index equation info
struct VirtualEqInfo {
    TensorEquation eq;
    std::string lhsTensorName;
    int lhsVirtualOffset;  // -1 for RHS-only equations
    std::map<std::pair<std::string, std::string>, std::set<int>> rhsVirtualRefs;  // (tensorName, virtualIndexName) -> offsets
    bool isRhsOnly;  // true if virtual indices only on RHS
};

// Build dependency graph for virtual-indexed equations within a timestep
struct DependencyGraph {
    std::vector<VirtualEqInfo> equations;
    std::vector<std::vector<int>> adjList;  // equation index -> list of dependent equation indices

    void addEquation(const VirtualEqInfo& info) {
        equations.push_back(info);
        adjList.resize(equations.size());
    }

    void buildEdges() {
        // For each equation, check if its RHS depends on another equation's LHS
        for (size_t i = 0; i < equations.size(); ++i) {
            for (size_t j = 0; j < equations.size(); ++j) {
                if (i == j) continue;

                const auto& eq_i = equations[i];
                const auto& eq_j = equations[j];

                // Check if equation j depends on equation i via virtual indices
                for (const auto& [key, offsets] : eq_j.rhsVirtualRefs) {
                    const auto& [tensorName, virtualIndexName] = key;
                    if (tensorName == eq_i.lhsTensorName) {
                        // j depends on i if j reads i's output offset
                        for (int offset : offsets) {
                            if (offset == eq_i.lhsVirtualOffset) {
                                adjList[i].push_back(j);
                                goto next_j;  // Found dependency, move to next j
                            }
                        }
                    }
                }

                // Also check for regular (non-virtual) tensor dependencies
                // If j's RHS uses i's LHS tensor (without virtual indices), it still depends on i
                // We need to check if j's equation references i's LHS tensor name
                // This is especially important for RHS-only equations
                if (referencesTensor(eq_j.eq, eq_i.lhsTensorName)) {
                    adjList[i].push_back(j);
                }

                next_j:;
            }
        }
    }

    // Helper to check if an equation's RHS references a specific tensor
    static bool referencesTensor(const TensorEquation& eq, const std::string& tensorName) {
        struct TensorChecker {
            const std::string& targetName;
            bool found = false;

            void visit(const Expr& expr) {
                if (found) return;
                std::visit([this](const auto& node) { (*this)(node); }, expr.node);
            }

            void operator()(const ExprTensorRef& ref) {
                if (ref.ref.name.name == targetName) {
                    found = true;
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
            void operator()(const ExprUnary& u) { visit(*u.operand); }
        } checker{tensorName, false};

        for (const auto& clause : eq.clauses) {
            checker.visit(*clause.expr);
            if (checker.found) return true;
        }
        return false;
    }

    // Topological sort using DFS
    std::vector<int> topologicalSort() {
        std::vector<int> result;
        std::vector<bool> visited(equations.size(), false);
        std::vector<bool> inStack(equations.size(), false);

        std::function<bool(int)> dfs = [&](int node) -> bool {
            if (inStack[node]) {
                throw std::runtime_error("Cyclic dependency detected in virtual-indexed equations");
            }
            if (visited[node]) return true;

            visited[node] = true;
            inStack[node] = true;

            for (int neighbor : adjList[node]) {
                if (!dfs(neighbor)) return false;
            }

            inStack[node] = false;
            result.push_back(node);
            return true;
        };

        for (size_t i = 0; i < equations.size(); ++i) {
            if (!visited[i]) {
                dfs(i);
            }
        }

        std::reverse(result.begin(), result.end());
        return result;
    }
};

// Substitute indices with SSA-style temporaries
// Mode B with intra-timestep dependencies: use SSA temporaries
TensorRef substituteIndicesSSA(const TensorRef& ref,
                               const std::map<std::string, int>& regularSubs,
                               const std::string& virtualIndexName,
                               const std::map<std::string, std::string>& tensorToTempName,
                               bool isLHS) {
    TensorRef result = ref;

    for (auto& idx : result.indices) {
        if (auto* vid = std::get_if<VirtualIndex>(&idx.value)) {
            // Replace virtual indices with 0
            // If virtualIndexName is empty, replace ALL virtual indices
            // If virtualIndexName is specified, only replace matching ones
            if (virtualIndexName.empty() || vid->name.name == virtualIndexName) {
                // This is our virtual index
                // Replace with appropriate temp name or slot
                // For now, we'll use temp tensor names, not slots
                // This reference will be rewritten to use temp tensors
                // Just mark it as 0 for now (we'll handle it differently)
                NumberLiteral num;
                num.text = "0";  // Placeholder
                num.loc = idx.loc;
                idx.value = num;
            }
        } else if (auto* id = std::get_if<Identifier>(&idx.value)) {
            // Regular identifier: look up in regularSubs
            auto it = regularSubs.find(id->name);
            if (it != regularSubs.end()) {
                NumberLiteral num;
                num.text = std::to_string(it->second);
                num.loc = idx.loc;
                idx.value = num;
            }
        }
    }

    // Replace tensor name with temp name if needed
    std::string tensorName = Environment::key(result);
    // Strip the indices part to get base tensor name
    size_t bracket = tensorName.find('[');
    std::string baseName = (bracket != std::string::npos) ? tensorName.substr(0, bracket) : tensorName;

    auto tempIt = tensorToTempName.find(baseName);
    if (tempIt != tensorToTempName.end()) {
        result.name.name = tempIt->second;
    }

    return result;
}

// Substitute indices in an expression
ExprPtr substituteIndicesInExprSSA(const ExprPtr& expr,
                                   const std::map<std::string, int>& regularSubs,
                                   const std::string& virtualIndexName,
                                   const std::map<std::string, std::string>& tensorToTempName);

struct SubstituterSSA {
    const std::map<std::string, int>& regularSubs;
    const std::string& virtualIndexName;
    const std::map<std::string, std::string>& tensorToTempName;

    ExprPtr operator()(const ExprTensorRef& ref) {
        auto result = std::make_shared<Expr>();
        result->loc = ref.ref.loc;
        result->node = ExprTensorRef{substituteIndicesSSA(ref.ref, regularSubs, virtualIndexName, tensorToTempName, false)};
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

ExprPtr substituteIndicesInExprSSA(const ExprPtr& expr,
                                   const std::map<std::string, int>& regularSubs,
                                   const std::string& virtualIndexName,
                                   const std::map<std::string, std::string>& tensorToTempName) {
    SubstituterSSA sub{regularSubs, virtualIndexName, tensorToTempName};
    return sub.visit(*expr);
}

} // anonymous namespace

std::vector<Statement> VirtualIndexPreprocessor::preprocessBatch(const std::vector<Statement>& statements, Environment& env) {
    // Group statements by virtual index variable
    std::map<std::string, std::vector<VirtualEqInfo>> groupsByVirtualIndex;

    bool debug = std::getenv("TL_DEBUG") != nullptr;

    if (debug) {
        std::cerr << "[VirtualIndexPreprocessor] Processing " << statements.size() << " statements\n";
    }

    for (const auto& st : statements) {
        if (!std::holds_alternative<TensorEquation>(st)) {
            continue;
        }
        const auto& eq = std::get<TensorEquation>(st);

        auto lhsVirtuals = findVirtualIndices(eq.lhs);
        auto rhsV = collectRhsVirtualIndices(eq);

        if (debug) {
            std::cerr << "[VirtualIndexPreprocessor]   Equation: " << Environment::key(eq.lhs) << " = ...\n";
            std::cerr << "[VirtualIndexPreprocessor]     LHS virtuals: " << lhsVirtuals.size() << "\n";
            std::cerr << "[VirtualIndexPreprocessor]     RHS virtual refs: " << rhsV.size() << "\n";
            for (const auto& [key, offsets] : rhsV) {
                const auto& [tensorName, virtualName] = key;
                std::cerr << "[VirtualIndexPreprocessor]       " << tensorName << "[*" << virtualName << "+offset], offsets: ";
                for (int off : offsets) std::cerr << off << " ";
                std::cerr << "\n";
            }
        }

        if (lhsVirtuals.empty() && !rhsV.empty()) {
            // This equation has virtual indices only on RHS - add to graph as RHS-only
            std::string lhsTensorName = Environment::key(eq.lhs);
            size_t bracket = lhsTensorName.find('[');
            std::string baseLhsName = (bracket != std::string::npos) ? lhsTensorName.substr(0, bracket) : lhsTensorName;

            if (debug) {
                std::cerr << "[VirtualIndexPreprocessor]     -> RHS-only equation, LHS base name: " << baseLhsName << "\n";
            }

            for (const auto& [key, offsets] : rhsV) {
                const auto& [tensorName, virtualName] = key;
                VirtualEqInfo info{eq, baseLhsName, -1, rhsV, true};
                groupsByVirtualIndex[virtualName].push_back(info);

                if (debug) {
                    std::cerr << "[VirtualIndexPreprocessor]     -> Added to group '" << virtualName << "'\n";
                }
            }
            continue;
        }

        if (lhsVirtuals.empty()) continue;

        if (lhsVirtuals.size() != 1) {
            throw std::runtime_error("Multiple virtual indices on LHS not yet supported");
        }

        std::string virtualIndexName = lhsVirtuals[0].first;
        int lhsOffset = lhsVirtuals[0].second;

        std::string lhsTensorName = Environment::key(eq.lhs);
        size_t bracket = lhsTensorName.find('[');
        std::string baseLhsName = (bracket != std::string::npos) ? lhsTensorName.substr(0, bracket) : lhsTensorName;

        if (debug) {
            std::cerr << "[VirtualIndexPreprocessor]     -> Virtual LHS: " << baseLhsName << "[*" << virtualIndexName << "+" << lhsOffset << "]\n";
        }

        VirtualEqInfo info{eq, baseLhsName, lhsOffset, rhsV, false};
        groupsByVirtualIndex[virtualIndexName].push_back(info);
    }

    std::vector<Statement> result;

    // Process each group independently
    for (auto& [virtualIndexName, eqInfos] : groupsByVirtualIndex) {
        if (eqInfos.empty()) continue;

        if (debug) {
            std::cerr << "[VirtualIndexPreprocessor] Processing group '" << virtualIndexName << "' with " << eqInfos.size() << " equations\n";
        }

        // Build dependency graph
        DependencyGraph graph;
        for (size_t i = 0; i < eqInfos.size(); ++i) {
            const auto& info = eqInfos[i];
            graph.addEquation(info);
            if (debug) {
                std::cerr << "[VirtualIndexPreprocessor]   Eq[" << i << "]: " << info.lhsTensorName;
                if (info.isRhsOnly) {
                    std::cerr << " (RHS-only)";
                } else {
                    std::cerr << "[*" << virtualIndexName << "+" << info.lhsVirtualOffset << "]";
                }
                std::cerr << "\n";
            }
        }
        graph.buildEdges();

        if (debug) {
            std::cerr << "[VirtualIndexPreprocessor]   Dependency edges:\n";
            for (size_t i = 0; i < graph.adjList.size(); ++i) {
                if (!graph.adjList[i].empty()) {
                    std::cerr << "[VirtualIndexPreprocessor]     " << i << " -> ";
                    for (int j : graph.adjList[i]) {
                        std::cerr << j << " ";
                    }
                    std::cerr << "\n";
                }
            }
        }

        // Topological sort
        std::vector<int> sortedIndices = graph.topologicalSort();

        if (debug) {
            std::cerr << "[VirtualIndexPreprocessor]   Topological order: ";
            for (int idx : sortedIndices) {
                std::cerr << idx << " ";
            }
            std::cerr << "\n";
        }

        // Determine iteration count (from first equation)
        int iterationCount = getIterationCount(virtualIndexName, env, eqInfos[0].eq);

        // Ensure all LHS tensors have minimum slots (skip RHS-only equations)
        for (const auto& info : eqInfos) {
            if (!info.isRhsOnly) {
                ensureMinimumVirtualSlots(info.eq, env, virtualIndexName, 1);
            }
        }

        // Expand timesteps
        for (int timestep = 0; timestep < iterationCount; ++timestep) {
            std::map<std::string, int> regularSubs {{ virtualIndexName, timestep }};

            if (debug) {
                std::cerr << "[VirtualIndexPreprocessor]   Timestep " << timestep << ":\n";
            }

            // Map tensor names to their SSA temporaries for this timestep
            std::map<std::string, std::string> tensorToTemp;
            for (int idx : sortedIndices) {
                const auto& info = eqInfos[idx];
                if (!info.isRhsOnly) {
                    std::string tempName = info.lhsTensorName + "_next_" + std::to_string(timestep);
                    tensorToTemp[info.lhsTensorName] = tempName;
                    if (debug) {
                        std::cerr << "[VirtualIndexPreprocessor]     Temp mapping: " << info.lhsTensorName << " -> " << tempName << "\n";
                    }
                }
            }

            // Generate equations in topological order
            for (int idx : sortedIndices) {
                const auto& info = eqInfos[idx];

                // Build RHS tensor map: *t reads from slot 0, *t+1 reads from temp
                std::map<std::string, std::string> rhsTensorMap;
                for (const auto& [key, offsets] : info.rhsVirtualRefs) {
                    const auto& [rhsTensorName, rhsVirtualName] = key;
                    for (int offset : offsets) {
                        // Check if this matches any tensor's LHS virtual offset
                        for (int checkIdx : sortedIndices) {
                            const auto& checkInfo = eqInfos[checkIdx];
                            if (!checkInfo.isRhsOnly && rhsTensorName == checkInfo.lhsTensorName &&
                                rhsVirtualName == virtualIndexName && offset == checkInfo.lhsVirtualOffset) {
                                // Reading *t+1: use temp if it exists
                                auto it = tensorToTemp.find(checkInfo.lhsTensorName);
                                if (it != tensorToTemp.end()) {
                                    rhsTensorMap[checkInfo.lhsTensorName] = it->second;
                                }
                            }
                        }
                    }
                }

                if (debug) {
                    std::cerr << "[VirtualIndexPreprocessor]       Processing eq[" << idx << "]: " << info.lhsTensorName;
                    if (info.isRhsOnly) {
                        std::cerr << " (RHS-only)";
                    }
                    std::cerr << "\n";
                    std::cerr << "[VirtualIndexPreprocessor]         RHS tensor map size: " << rhsTensorMap.size() << "\n";
                    for (const auto& [tn, temp] : rhsTensorMap) {
                        std::cerr << "[VirtualIndexPreprocessor]           " << tn << " -> " << temp << "\n";
                    }
                }

                if (info.isRhsOnly) {
                    // RHS-only equation: substitute indices in both LHS and RHS
                    TensorEquation expandedEq = info.eq;

                    // Substitute LHS indices (especially regular indices like 't')
                    expandedEq.lhs = substituteIndicesSSA(info.eq.lhs, regularSubs, virtualIndexName, std::map<std::string, std::string>{}, false);

                    // Substitute RHS indices
                    for (size_t clauseIdx = 0; clauseIdx < expandedEq.clauses.size(); ++clauseIdx) {
                        expandedEq.clauses[clauseIdx].expr =
                            substituteIndicesInExprSSA(info.eq.clauses[clauseIdx].expr, regularSubs, virtualIndexName, rhsTensorMap);
                        if (info.eq.clauses[clauseIdx].guard) {
                            expandedEq.clauses[clauseIdx].guard =
                                substituteIndicesInExprSSA(*info.eq.clauses[clauseIdx].guard, regularSubs, virtualIndexName, rhsTensorMap);
                        }
                    }
                    if (debug) {
                        std::cerr << "[VirtualIndexPreprocessor]         -> Generated RHS-only: " << Environment::key(expandedEq.lhs) << "\n";
                    }
                    result.push_back(expandedEq);
                } else {
                    // Regular equation with virtual index on LHS: write to temp
                    std::string tempName = tensorToTemp[info.lhsTensorName];
                    TensorEquation writeEq = info.eq;
                    writeEq.lhs.name.name = tempName;

                    // MODE B: Remove virtual indices entirely from temp tensors
                    std::vector<Index> newIndices;
                    for (auto& idx : writeEq.lhs.indices) {
                        if (!std::holds_alternative<VirtualIndex>(idx.value)) {
                            newIndices.push_back(idx);
                        }
                    }
                    writeEq.lhs.indices = newIndices;

                    for (size_t clauseIdx = 0; clauseIdx < writeEq.clauses.size(); ++clauseIdx) {
                        writeEq.clauses[clauseIdx].expr =
                            substituteIndicesInExprSSA(info.eq.clauses[clauseIdx].expr, regularSubs, virtualIndexName, rhsTensorMap);
                        if (info.eq.clauses[clauseIdx].guard) {
                            writeEq.clauses[clauseIdx].guard =
                                substituteIndicesInExprSSA(*info.eq.clauses[clauseIdx].guard, regularSubs, virtualIndexName, rhsTensorMap);
                        }
                    }
                    if (debug) {
                        std::cerr << "[VirtualIndexPreprocessor]         -> Generated write to temp: " << tempName << "\n";
                    }
                    result.push_back(writeEq);
                }
            }

            if (debug) {
                std::cerr << "[VirtualIndexPreprocessor]     Generated " << result.size() << " equations so far\n";
            }

            // After all equations in this timestep, copy temps back to main tensors
            for (int idx : sortedIndices) {
                const auto& info = eqInfos[idx];

                // Only copy back non-RHS-only equations
                if (info.isRhsOnly) continue;

                std::string tempName = tensorToTemp[info.lhsTensorName];

                TensorEquation copyEq;
                copyEq.projection = info.eq.projection;
                copyEq.loc = info.eq.loc;

                copyEq.lhs = info.eq.lhs;
                // MODE B: Substitute virtual indices with 0 (main tensors keep the dimension)
                for (auto& idx : copyEq.lhs.indices) {
                    if (std::holds_alternative<VirtualIndex>(idx.value)) {
                        NumberLiteral num;
                        num.text = "0";
                        num.loc = idx.loc;
                        idx.value = num;
                    }
                }

                auto rhsRef = std::make_shared<Expr>();
                rhsRef->loc = info.eq.lhs.loc;
                TensorRef readRef = info.eq.lhs;
                readRef.name.name = tempName;
                // MODE B: Remove virtual indices entirely from RHS read reference
                std::vector<Index> newReadIndices;
                for (auto& idx : readRef.indices) {
                    if (!std::holds_alternative<VirtualIndex>(idx.value)) {
                        newReadIndices.push_back(idx);
                    }
                }
                readRef.indices = newReadIndices;
                rhsRef->node = ExprTensorRef{readRef};

                GuardedClause copyClause;
                copyClause.expr = rhsRef;
                copyEq.clauses.push_back(copyClause);

                result.push_back(copyEq);
            }
        }
    }

    return result;
}

bool VirtualIndexPreprocessor::shouldPreprocess(const Statement& st, const Environment& env) const {
    if (!std::holds_alternative<TensorEquation>(st)) {
        return false;
    }
    const auto& eq = std::get<TensorEquation>(st);

    if (hasVirtualIndices(eq.lhs.indices)) {
        return true;
    }

    const auto rhsV = collectRhsVirtualIndices(eq);
    return !rhsV.empty();
}

std::vector<Statement> VirtualIndexPreprocessor::preprocess(const Statement& st, Environment& env) {
    if (!std::holds_alternative<TensorEquation>(st)) {
        return {st};
    }

    const auto& eq = std::get<TensorEquation>(st);

    // Step 1: Find virtual index info from LHS
    auto lhsVirtuals = findVirtualIndices(eq.lhs);
    if (lhsVirtuals.empty()) {
        // No virtual indices in LHS - check if there are any in RHS
        auto rhsV = collectRhsVirtualIndices(eq);
        if (rhsV.empty()) return {st};

        // RHS-only virtual indices: queries using final state
        // Just map to slot 0
        std::map<std::string, int> regularSubs;
        std::map<std::string, std::string> emptyTensorMap;
        TensorEquation concreteEq = eq;

        for (size_t clauseIdx = 0; clauseIdx < concreteEq.clauses.size(); ++clauseIdx) {
            concreteEq.clauses[clauseIdx].expr =
                substituteIndicesInExprSSA(eq.clauses[clauseIdx].expr, regularSubs, "", emptyTensorMap);
            if (eq.clauses[clauseIdx].guard) {
                concreteEq.clauses[clauseIdx].guard =
                    substituteIndicesInExprSSA(*eq.clauses[clauseIdx].guard, regularSubs, "", emptyTensorMap);
            }
        }

        return {concreteEq};
    }

    // For now, we only support single virtual index on LHS
    if (lhsVirtuals.size() != 1) {
        throw std::runtime_error("Multiple virtual indices on LHS not yet supported");
    }

    std::string virtualIndexName = lhsVirtuals[0].first;
    int lhsOffset = lhsVirtuals[0].second;

    // Step 2: Find regular indices in RHS that match the virtual index name
    auto regularIndices = findRegularIndices(eq);

    if (regularIndices.find(virtualIndexName) == regularIndices.end()) {
        throw std::runtime_error(
            "Virtual index '" + virtualIndexName + "' must appear as regular index in RHS to drive iteration");
    }

    // Step 3: Determine iteration count
    int iterationCount = getIterationCount(virtualIndexName, env, eq);

    // Step 4: MODE B with SSA - ensure tensor exists with proper shape
    ensureMinimumVirtualSlots(eq, env, virtualIndexName, 1);

    // Step 5: Expand into concrete statements (Mode B with SSA temporaries)
    std::vector<Statement> result;

    for (int i = 0; i < iterationCount; ++i) {
        std::map<std::string, int> regularSubs {{ virtualIndexName, i }};

        // Create SSA-style temporary name for the next state
        std::string lhsTensorName = Environment::key(eq.lhs);
        size_t bracket = lhsTensorName.find('[');
        std::string baseLhsName = (bracket != std::string::npos) ? lhsTensorName.substr(0, bracket) : lhsTensorName;

        std::string tempName = baseLhsName + "_next_" + std::to_string(i);

        // Map: read from State[0] for *t, write to temp for *t+1
        std::map<std::string, std::string> tensorToTempForLHS;
        tensorToTempForLHS[baseLhsName] = tempName;

        // For RHS: *t reads from State[0], *t+1 reads from temp (if it exists)
        auto rhsV = collectRhsVirtualIndices(eq);
        std::map<std::string, std::string> tensorToTempForRHS;

        // Check if RHS uses any *t+1 references to other tensors
        // Those should read from their _next temps
        for (const auto& [key, offsets] : rhsV) {
            const auto& [rhsTensorName, rhsVirtualName] = key;
            if (rhsTensorName == baseLhsName && rhsVirtualName == virtualIndexName) {
                for (int off : offsets) {
                    if (off == lhsOffset) {
                        // RHS uses *t+1 of the same tensor we're writing
                        // This means it's reading the just-computed value
                        // Use the temp name
                        tensorToTempForRHS[baseLhsName] = tempName;
                    }
                }
            }
        }

        // Statement 1: Compute into temp
        TensorEquation writeEq = eq;
        writeEq.lhs.name.name = tempName;
        // MODE B: Remove virtual indices entirely from temp tensor LHS
        std::vector<Index> newWriteIndices;
        for (auto& idx : writeEq.lhs.indices) {
            if (!std::holds_alternative<VirtualIndex>(idx.value)) {
                newWriteIndices.push_back(idx);
            }
        }
        writeEq.lhs.indices = newWriteIndices;

        for (size_t clauseIdx = 0; clauseIdx < writeEq.clauses.size(); ++clauseIdx) {
            writeEq.clauses[clauseIdx].expr =
                substituteIndicesInExprSSA(eq.clauses[clauseIdx].expr, regularSubs, virtualIndexName, tensorToTempForRHS);
            if (eq.clauses[clauseIdx].guard) {
                writeEq.clauses[clauseIdx].guard =
                    substituteIndicesInExprSSA(*eq.clauses[clauseIdx].guard, regularSubs, virtualIndexName, tensorToTempForRHS);
            }
        }
        result.push_back(writeEq);

        // Statement 2: Copy from temp back to main tensor at slot 0
        TensorEquation copyEq;
        copyEq.projection = eq.projection;
        copyEq.loc = eq.loc;

        // LHS: original tensor at slot 0 (main tensors keep the dimension)
        copyEq.lhs = eq.lhs;
        // MODE B: Substitute virtual indices with 0 (main tensors keep the dimension)
        for (auto& idx : copyEq.lhs.indices) {
            if (std::holds_alternative<VirtualIndex>(idx.value)) {
                NumberLiteral num;
                num.text = "0";
                num.loc = idx.loc;
                idx.value = num;
            }
        }

        // RHS: temp tensor
        auto rhsRef = std::make_shared<Expr>();
        rhsRef->loc = eq.lhs.loc;
        TensorRef readRef = eq.lhs;
        readRef.name.name = tempName;
        // MODE B: Remove virtual indices entirely from copy-back RHS
        std::vector<Index> newCopyRhsIndices;
        for (auto& idx : readRef.indices) {
            if (!std::holds_alternative<VirtualIndex>(idx.value)) {
                newCopyRhsIndices.push_back(idx);
            }
        }
        readRef.indices = newCopyRhsIndices;
        rhsRef->node = ExprTensorRef{readRef};

        GuardedClause copyClause;
        copyClause.expr = rhsRef;
        copyEq.clauses.push_back(copyClause);

        result.push_back(copyEq);
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
    // Collect all candidate tensors that use this index
    struct TensorRefFinder {
        const std::string& targetIndex;
        std::vector<TensorRef> candidates;

        void visit(const Expr& expr) {
            std::visit([this](const auto& node) { this->operator()(node); }, expr.node);
        }

        void operator()(const ExprTensorRef& ref) {
            for (const auto& idx : ref.ref.indices) {
                if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                    if (id->name == targetIndex) {
                        candidates.push_back(ref.ref);
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
            visit(*b.rhs);
        }
        void operator()(const ExprUnary& u) { visit(*u.operand); }
    };

    for (const auto& clause : eq.clauses) {
        TensorRefFinder finder{indexName, {}};
        finder.visit(*clause.expr);

        // Try each candidate until we find one that exists in the environment
        for (const auto& ref : finder.candidates) {
            if (env.has(ref)) {
                const Tensor& t = env.lookup(ref);

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
    }

    // If we can't find any tensor, try default of 10 or throw error
    return 10;
}

void VirtualIndexPreprocessor::preallocateStorage(const TensorEquation& eq, Environment& env,
                                                   const std::string& virtualIndexName,
                                                   int lhsOffset, int iterationCount) {
    // Mode B: Deprecated
}

void VirtualIndexPreprocessor::ensureMinimumVirtualSlots(const TensorEquation& eq, Environment& env,
                                                          const std::string& virtualIndexName,
                                                          int minSlots) {
    std::string lhsTensorName = Environment::key(eq.lhs);

    int virtualDim = -1;
    for (size_t i = 0; i < eq.lhs.indices.size(); ++i) {
        if (std::holds_alternative<VirtualIndex>(eq.lhs.indices[i].value)) {
            virtualDim = static_cast<int>(i);
            break;
        }
    }

    if (virtualDim == -1) {
        return;
    }

    if (!env.has(lhsTensorName)) {
        return;
    }

    Tensor existingTensor = env.lookup(lhsTensorName);

    if (virtualDim >= existingTensor.dim()) {
        std::vector<int64_t> newShape;
        for (int d = 0; d < existingTensor.dim(); ++d) {
            newShape.push_back(existingTensor.size(d));
        }
        while (static_cast<int>(newShape.size()) <= virtualDim) {
            newShape.push_back(minSlots);
        }

        Tensor newTensor = torch::zeros(newShape, existingTensor.options());

        std::vector<torch::indexing::TensorIndex> sliceIndices;
        for (int d = 0; d < existingTensor.dim(); ++d) {
            sliceIndices.push_back(torch::indexing::Slice(0, existingTensor.size(d)));
        }
        newTensor.index(sliceIndices).copy_(existingTensor);

        env.bind(lhsTensorName, newTensor);
    } else if (existingTensor.size(virtualDim) < minSlots) {
        std::vector<int64_t> newShape;
        for (int d = 0; d < existingTensor.dim(); ++d) {
            if (d == virtualDim) {
                newShape.push_back(minSlots);
            } else {
                newShape.push_back(existingTensor.size(d));
            }
        }

        Tensor newTensor = torch::zeros(newShape, existingTensor.options());

        std::vector<torch::indexing::TensorIndex> sliceIndices;
        for (int d = 0; d < existingTensor.dim(); ++d) {
            sliceIndices.push_back(torch::indexing::Slice(0, existingTensor.size(d)));
        }
        newTensor.index(sliceIndices).copy_(existingTensor);

        env.bind(lhsTensorName, newTensor);
    }
}

} // namespace tl
