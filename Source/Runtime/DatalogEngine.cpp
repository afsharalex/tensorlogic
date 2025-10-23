#include "TL/Runtime/DatalogEngine.hpp"
#include "TL/vm.hpp"
#include <sstream>
#include <unordered_set>
#include <optional>
#include <cctype>
#include <functional>

namespace tl {

DatalogEngine::DatalogEngine(Environment& env, std::ostream* out)
    : env_(env)
    , output_stream_(out)
{}

bool DatalogEngine::addFact(const DatalogFact& fact) {
    bool inserted = env_.addFact(fact);
    if (inserted) {
        closure_dirty_ = true;
        if (debug_) {
            std::ostringstream oss;
            oss << "Added fact: " << fact.relation.name << "(";
            for (size_t i = 0; i < fact.constants.size(); ++i) {
                if (i) oss << ", ";
                oss << fact.constants[i].text;
            }
            oss << ")";
            debugLog(oss.str());
        }
    }
    return inserted;
}

void DatalogEngine::addRule(const DatalogRule& rule) {
    rules_.push_back(rule);
    closure_dirty_ = true;
    if (debug_) {
        debugLog("Registered Datalog rule");
    }
}

void DatalogEngine::saturate() {
    if (!closure_dirty_ || rules_.empty()) return;

    size_t totalNew = 0;
    size_t iter = 0;
    while (true) {
        size_t roundNew = 0;
        for (const auto& r : rules_) {
            roundNew += applyRule(r);
        }
        totalNew += roundNew;
        ++iter;
        if (roundNew == 0) break;
    }

    if (debug_) {
        debugLog("Rule saturation finished after fixpoint.");
    }
    closure_dirty_ = false;
}

size_t DatalogEngine::applyRule(const DatalogRule& rule) {
    // Collect body atoms and conditions
    std::vector<DatalogAtom> bodyAtoms;
    std::vector<DatalogCondition> conditions;
    bodyAtoms.reserve(rule.body.size());
    for (const auto& el : rule.body) {
        if (const auto* a = std::get_if<DatalogAtom>(&el)) bodyAtoms.push_back(*a);
        else if (const auto* c = std::get_if<DatalogCondition>(&el)) conditions.push_back(*c);
    }
    if (bodyAtoms.empty()) return 0;

    size_t newCount = 0;

    // Depth-first join over body atoms
    std::unordered_map<std::string, std::string> binding;
    std::function<void(size_t)> dfs = [&](size_t idx) {
        if (idx == bodyAtoms.size()) {
            // Evaluate conditions as filters
            for (const auto& cond : conditions) {
                if (!evalCondition(cond, binding)) return; // reject this binding
            }
            // Build head tuple
            std::vector<std::string> headTuple;
            headTuple.reserve(rule.head.terms.size());
            for (const auto& t : rule.head.terms) {
                if (std::holds_alternative<StringLiteral>(t)) {
                    headTuple.push_back(std::get<StringLiteral>(t).text);
                } else {
                    const std::string& vn = std::get<Identifier>(t).name;
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

        const DatalogAtom& atom = bodyAtoms[idx];
        const auto& tuples = env_.facts(atom.relation.name);
        for (const auto& tup : tuples) {
            if (tup.size() != atom.terms.size()) continue;
            // Local modifications to binding; keep a list to rollback
            std::vector<std::string> assignedVars;
            bool ok = true;
            for (size_t i = 0; i < atom.terms.size(); ++i) {
                const auto& term = atom.terms[i];
                const std::string& val = tup[i];
                if (std::holds_alternative<StringLiteral>(term)) {
                    if (std::get<StringLiteral>(term).text != val) { ok = false; break; }
                } else {
                    const std::string& vn = std::get<Identifier>(term).name;
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
            for (const auto& vn : assignedVars) binding.erase(vn);
        }
    };

    dfs(0);
    return newCount;
}

void DatalogEngine::query(const Query& q, std::ostream& out) {
    // Only handle Datalog queries here; tensor queries handled by VM
    if (std::holds_alternative<TensorRef>(q.target)) {
        // This shouldn't happen - tensor queries should be handled by VM
        throw std::runtime_error("DatalogEngine::query called with TensorRef query");
    }

    const auto& atom = std::get<DatalogAtom>(q.target);
    execDatalogQuery(atom, q.body, out);
}

void DatalogEngine::execDatalogQuery(const DatalogAtom& atom,
                                     const std::vector<std::variant<DatalogAtom, DatalogCondition>>& body,
                                     std::ostream& out) {
    // If this is a conjunctive Datalog query with optional comparisons, evaluate via join
    if (!body.empty()) {
        // Separate atoms and conditions
        std::vector<DatalogAtom> atoms;
        std::vector<DatalogCondition> conditions;
        atoms.reserve(body.size());
        for (const auto& el : body) {
            if (const auto* a = std::get_if<DatalogAtom>(&el)) atoms.push_back(*a);
            else if (const auto* c = std::get_if<DatalogCondition>(&el)) conditions.push_back(*c);
        }
        if (atoms.empty()) {
            out << "None" << std::endl;
            return;
        }

        // Determine variable output order across atoms by first appearance
        std::vector<std::string> varNames;
        std::unordered_set<std::string> seen;
        for (const auto& a : atoms) {
            for (const auto& t : a.terms) {
                if (const auto* id = std::get_if<Identifier>(&t)) {
                    const std::string& vn = id->name;
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
                for (const auto& cond : conditions) {
                    if (!evalCondition(cond, binding)) return;
                }
                if (varNames.empty()) {
                    out << "True" << std::endl;
                    anyPrinted = true;
                    return;
                }
                if (varNames.size() == 1) {
                    out << binding[varNames[0]] << std::endl;
                    anyPrinted = true;
                } else {
                    for (size_t i = 0; i < varNames.size(); ++i) {
                        if (i) out << ", ";
                        out << binding[varNames[i]];
                    }
                    out << std::endl;
                    anyPrinted = true;
                }
                return;
            }
            const DatalogAtom& a = atoms[idx];
            const auto& tuples = env_.facts(a.relation.name);
            for (const auto& tup : tuples) {
                if (tup.size() != a.terms.size()) continue;
                std::vector<std::string> assigned;
                bool ok = true;
                for (size_t i = 0; i < a.terms.size(); ++i) {
                    const auto& term = a.terms[i];
                    const std::string& val = tup[i];
                    if (std::holds_alternative<StringLiteral>(term)) {
                        if (std::get<StringLiteral>(term).text != val) { ok = false; break; }
                    } else {
                        const std::string& vn = std::get<Identifier>(term).name;
                        auto it = binding.find(vn);
                        if (it == binding.end()) { binding.emplace(vn, val); assigned.push_back(vn); }
                        else if (it->second != val) { ok = false; break; }
                    }
                }
                if (ok) dfs(idx + 1);
                for (const auto& vn : assigned) binding.erase(vn);
            }
        };

        dfs(0);
        if (!anyPrinted) {
            // Ground conjunctive query with no satisfying assignment
            if (varNames.empty()) {
                out << "False" << std::endl;
            } else {
                out << "None" << std::endl;
            }
        }
        return;
    }

    // Simple single-atom query
    const std::string rel = atom.relation.name;
    if (debug_) {
        std::ostringstream oss;
        oss << "Query over Datalog atom: " << rel << "(";
        for (size_t i = 0; i < atom.terms.size(); ++i) {
            if (i) oss << ", ";
            if (std::holds_alternative<Identifier>(atom.terms[i]))
                oss << std::get<Identifier>(atom.terms[i]).name;
            else
                oss << std::get<StringLiteral>(atom.terms[i]).text;
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
            const std::string& vname = std::get<Identifier>(atom.terms[i]).name;
            if (!firstPos.count(vname)) {
                firstPos[vname] = static_cast<int>(i);
                varPositions.push_back(static_cast<int>(i));
                varNames.push_back(vname);
            }
        } else {
            constants[i] = std::get<StringLiteral>(atom.terms[i]).text;
        }
    }

    const auto& tuples = env_.facts(rel);
    auto matchesTuple = [&](const std::vector<std::string>& tuple) -> bool {
        if (tuple.size() != atom.terms.size()) return false;
        // Check constants
        for (size_t i = 0; i < constants.size(); ++i) {
            if (constants[i].has_value() && tuple[i] != *constants[i]) return false;
        }
        // Check repeated vars consistency
        std::unordered_map<std::string, std::string> bind;
        for (size_t i = 0; i < atom.terms.size(); ++i) {
            if (std::holds_alternative<Identifier>(atom.terms[i])) {
                const std::string& vn = std::get<Identifier>(atom.terms[i]).name;
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
        for (const auto& tup : tuples) { if (matchesTuple(tup)) { any = true; break; } }
        out << (any ? "True" : "False") << std::endl;
        return;
    }

    // Variable bindings: print each matching binding
    bool anyPrinted = false;
    for (const auto& tup : tuples) {
        if (!matchesTuple(tup)) continue;
        if (varNames.size() == 1) {
            out << tup[varPositions[0]] << std::endl;
            anyPrinted = true;
        } else {
            // Print comma-separated values for the variables in first-appearance order
            for (size_t i = 0; i < varNames.size(); ++i) {
                if (i) out << ", ";
                out << tup[varPositions[i]];
            }
            out << std::endl;
            anyPrinted = true;
        }
    }
    if (!anyPrinted) {
        out << "None" << std::endl;
    }
}

bool DatalogEngine::evalExprBinding(const ExprPtr& e,
                                    const std::unordered_map<std::string, std::string>& binding,
                                    std::string& outStr,
                                    double& outNum,
                                    bool& isNumeric) const {
    if (!e) return false;
    const Expr& ex = *e;

    if (const auto* num = std::get_if<ExprNumber>(&ex.node)) {
        outStr = num->literal.text;
        try { outNum = std::stod(outStr); isNumeric = true; } catch (...) { isNumeric = false; }
        return true;
    }

    if (const auto* str = std::get_if<ExprString>(&ex.node)) {
        outStr = str->literal.text;
        try { outNum = std::stod(outStr); isNumeric = true; } catch (...) { isNumeric = false; }
        return true;
    }

    if (const auto* tr = std::get_if<ExprTensorRef>(&ex.node)) {
        // Treat lowercase scalar identifiers (no indices) as Datalog variables
        const std::string& name = tr->ref.name.name;
        const bool isVar = tr->ref.indices.empty() && !name.empty() &&
                          std::islower(static_cast<unsigned char>(name[0])) != 0;
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

    if (const auto* paren = std::get_if<ExprParen>(&ex.node)) {
        return evalExprBinding(paren->inner, binding, outStr, outNum, isNumeric);
    }

    if (const auto* bin = std::get_if<ExprBinary>(&ex.node)) {
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

bool DatalogEngine::evalCondition(const DatalogCondition& cond,
                                  const std::unordered_map<std::string, std::string>& binding) const {
    std::string ls, rs;
    double ln = 0, rn = 0;
    bool lnum = false, rnum = false;

    if (!evalExprBinding(cond.lhs, binding, ls, ln, lnum)) return false;
    if (!evalExprBinding(cond.rhs, binding, rs, rn, rnum)) return false;

    auto doStrCmp = [&](const std::string& op) {
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

void DatalogEngine::debugLog(const std::string& msg) const {
    if (debug_) {
        (*output_stream_) << "[DatalogEngine] " << msg << std::endl;
    }
}

} // namespace tl
