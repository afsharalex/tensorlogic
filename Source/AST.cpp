#include "TL/AST.hpp"
#include <sstream>

namespace tl {

std::string toString(const Identifier& id) { return id.name; }

static std::string indexToString(const Index& idx) {
    if (std::holds_alternative<Identifier>(idx.value)) {
        return std::get<Identifier>(idx.value).name;
    } else if (std::holds_alternative<NumberLiteral>(idx.value)) {
        return std::get<NumberLiteral>(idx.value).text;
    } else {
        // VirtualIndex
        const auto& vidx = std::get<VirtualIndex>(idx.value);
        std::string result = "*" + vidx.name.name;
        if (vidx.offset > 0) result += "+" + std::to_string(vidx.offset);
        else if (vidx.offset < 0) result += std::to_string(vidx.offset);
        return result;
    }
}

static std::string sliceToString(const Slice& slice) {
    std::ostringstream oss;
    if (slice.start.has_value()) {
        oss << slice.start->text;
    }
    oss << ":";
    if (slice.end.has_value()) {
        oss << slice.end->text;
    }
    if (slice.step.has_value()) {
        oss << ":" << slice.step->text;
    }
    return oss.str();
}

static std::string indexOrSliceToString(const IndexOrSlice& ios) {
    if (std::holds_alternative<Index>(ios.value)) {
        return indexToString(std::get<Index>(ios.value));
    } else {
        return sliceToString(std::get<Slice>(ios.value));
    }
}

std::string toString(const TensorRef& ref) {
    std::ostringstream oss;
    oss << ref.name.name;
    if (!ref.indices.empty()) {
        oss << "[";
        for (size_t i = 0; i < ref.indices.size(); ++i) {
            if (i) oss << ",";
            oss << indexOrSliceToString(ref.indices[i]);
        }
        oss << "]";
    }
    return oss.str();
}

std::string toString(const Expr& e) {
    struct V {
        std::string operator()(const ExprTensorRef& n) const { return toString(n.ref); }
        std::string operator()(const ExprNumber& n) const { return n.literal.text; }
        std::string operator()(const ExprString& s) const { return '"' + s.literal.text + '"'; }
        std::string operator()(const ExprParen& p) const { return '(' + toString(*p.inner) + ')'; }
        std::string operator()(const ExprList& lst) const {
            std::ostringstream oss; oss << '[';
            for (size_t i = 0; i < lst.elements.size(); ++i) {
                if (i) oss << ",";
                oss << toString(*lst.elements[i]);
            }
            oss << ']';
            return oss.str();
        }
        std::string operator()(const ExprCall& c) const {
            std::ostringstream oss; oss << c.func.name << '(';
            for (size_t i = 0; i < c.args.size(); ++i) {
                if (i) oss << ",";
                oss << toString(*c.args[i]);
            }
            oss << ')';
            return oss.str();
        }
        std::string operator()(const ExprBinary& b) const {
            switch (b.op) {
                case ExprBinary::Op::Add: return toString(*b.lhs) + "+" + toString(*b.rhs);
                case ExprBinary::Op::Sub: return toString(*b.lhs) + "-" + toString(*b.rhs);
                case ExprBinary::Op::Mul: return toString(*b.lhs) + toString(*b.rhs); // implicit multiplication
                case ExprBinary::Op::Div: return toString(*b.lhs) + "/" + toString(*b.rhs);
                case ExprBinary::Op::Mod: return toString(*b.lhs) + "%" + toString(*b.rhs);
                case ExprBinary::Op::Pow: return toString(*b.lhs) + "^" + toString(*b.rhs);
                case ExprBinary::Op::Lt: return toString(*b.lhs) + "<" + toString(*b.rhs);
                case ExprBinary::Op::Le: return toString(*b.lhs) + "<=" + toString(*b.rhs);
                case ExprBinary::Op::Gt: return toString(*b.lhs) + ">" + toString(*b.rhs);
                case ExprBinary::Op::Ge: return toString(*b.lhs) + ">=" + toString(*b.rhs);
                case ExprBinary::Op::Eq: return toString(*b.lhs) + "==" + toString(*b.rhs);
                case ExprBinary::Op::Ne: return toString(*b.lhs) + "!=" + toString(*b.rhs);
                case ExprBinary::Op::And: return toString(*b.lhs) + " and " + toString(*b.rhs);
                case ExprBinary::Op::Or: return toString(*b.lhs) + " or " + toString(*b.rhs);
            }
            return toString(*b.lhs) + toString(*b.rhs);
        }
        std::string operator()(const ExprUnary& u) const {
            switch (u.op) {
                case ExprUnary::Op::Neg: return "-" + toString(*u.operand);
                case ExprUnary::Op::Not: return "not " + toString(*u.operand);
            }
            return toString(*u.operand);
        }
    } v;
    return std::visit(v, e.node);
}

static std::string datalogAtomToString(const DatalogAtom& a) {
    std::ostringstream oss;
    oss << a.relation.name << '(';
    for (size_t i = 0; i < a.terms.size(); ++i) {
        if (i) oss << ',';
        if (std::holds_alternative<Identifier>(a.terms[i])) {
            oss << std::get<Identifier>(a.terms[i]).name;
        } else {
            oss << std::get<StringLiteral>(a.terms[i]).text;
        }
    }
    oss << ')';
    return oss.str();
}

std::string toString(const Statement& st) {
    struct V2 {
        std::string operator()(const TensorEquation& eq) const {
            std::ostringstream oss;
            oss << toString(eq.lhs) << " " << eq.projection << " ";
            for (size_t i = 0; i < eq.clauses.size(); ++i) {
                if (i) oss << " | ";
                oss << toString(*eq.clauses[i].expr);
                if (eq.clauses[i].guard) {
                    oss << " : " << toString(**eq.clauses[i].guard);
                }
            }
            return oss.str();
        }
        std::string operator()(const FileOperation& fo) const {
            std::ostringstream oss;
            if (fo.lhsIsTensor) {
                oss << toString(fo.tensor) << " = \"" << fo.file.text << "\"";
            } else {
                oss << "\"" << fo.file.text << "\" = " << toString(fo.tensor);
            }
            return oss.str();
        }
        std::string operator()(const Query& q) const {
            std::ostringstream oss;

            // For Datalog conjunctive queries, the first atom appears in both `target` and `body[0]`
            // To avoid duplication, we print only the body if it's non-empty AND target is a DatalogAtom
            if (std::holds_alternative<DatalogAtom>(q.target) && !q.body.empty()) {
                // Print all body elements (which includes the target as first element)
                for (size_t i = 0; i < q.body.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << bodyElemToString(q.body[i]);
                }
            } else {
                // Tensor query or simple query without conjunction
                if (std::holds_alternative<TensorRef>(q.target)) {
                    oss << toString(std::get<TensorRef>(q.target));
                } else {
                    oss << datalogAtomToString(std::get<DatalogAtom>(q.target));
                }
                // Print any additional body elements (shouldn't happen for tensor queries)
                if (!q.body.empty()) {
                    for (const auto& elem : q.body) {
                        oss << ", " << bodyElemToString(elem);
                    }
                }
            }

            oss << "?";
            return oss.str();
        }
        std::string operator()(const DatalogFact& f) const {
            std::ostringstream oss; oss << f.relation.name << '(';
            for (size_t i = 0; i < f.constants.size(); ++i) {
                if (i) oss << ",";
                oss << f.constants[i].text; // constants stored as strings without quotes
            }
            oss << ')';
            return oss.str();
        }
        static std::string bodyElemToString(const std::variant<DatalogAtom, DatalogNegation, DatalogCondition>& e) {
            if (std::holds_alternative<DatalogAtom>(e)) return datalogAtomToString(std::get<DatalogAtom>(e));
            if (std::holds_alternative<DatalogNegation>(e)) return std::string("not ") + datalogAtomToString(std::get<DatalogNegation>(e).atom);
            const auto& c = std::get<DatalogCondition>(e);
            return toString(*c.lhs) + " " + c.op + " " + toString(*c.rhs);
        }
        std::string operator()(const DatalogRule& r) const {
            std::ostringstream oss;
            oss << datalogAtomToString(r.head) << " <- ";
            for (size_t i = 0; i < r.body.size(); ++i) {
                if (i) oss << ',';
                oss << bodyElemToString(r.body[i]);
            }
            return oss.str();
        }
        std::string operator()(const FixedPointLoop& loop) const {
            std::ostringstream oss;
            oss << "FixedPointLoop(" << loop.monitoredTensor << "): ";
            oss << toString(loop.equation.lhs) << " " << loop.equation.projection << " ";
            for (size_t i = 0; i < loop.equation.clauses.size(); ++i) {
                if (i) oss << " | ";
                oss << toString(*loop.equation.clauses[i].expr);
                if (loop.equation.clauses[i].guard) {
                    oss << " : " << toString(**loop.equation.clauses[i].guard);
                }
            }
            return oss.str();
        }
    } v;
    return std::visit(v, st);
}

} // namespace tl
