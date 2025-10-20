#include "TL/AST.hpp"
#include <sstream>

namespace tl {

std::string toString(const Identifier& id) { return id.name; }

static std::string indexToString(const Index& idx) {
    if (std::holds_alternative<Identifier>(idx.value)) return std::get<Identifier>(idx.value).name;
    return std::get<NumberLiteral>(idx.value).text;
}

std::string toString(const TensorRef& ref) {
    std::ostringstream oss;
    oss << ref.name.name;
    if (!ref.indices.empty()) {
        oss << "[";
        for (size_t i = 0; i < ref.indices.size(); ++i) {
            if (i) oss << ",";
            oss << indexToString(ref.indices[i]);
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
            }
            return toString(*b.lhs) + toString(*b.rhs);
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
            std::ostringstream oss; oss << toString(eq.lhs) << " " << eq.projection << " " << toString(*eq.rhs);
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
            if (std::holds_alternative<TensorRef>(q.target)) {
                return toString(std::get<TensorRef>(q.target)) + "?";
            } else {
                return datalogAtomToString(std::get<DatalogAtom>(q.target)) + "?";
            }
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
        static std::string bodyElemToString(const std::variant<DatalogAtom, DatalogCondition>& e) {
            if (std::holds_alternative<DatalogAtom>(e)) return datalogAtomToString(std::get<DatalogAtom>(e));
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
    } v;
    return std::visit(v, st);
}

} // namespace tl
