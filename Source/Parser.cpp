#include "TL/Parser.hpp"
#include <fstream>
#include <sstream>
#include "TL/Lexer.hpp"

namespace tl {

namespace {

using tl::lex::Token;
using tl::lex::TokenStream;

class Parser {
public:
    explicit Parser(const std::string_view src) : toks_(src) {
        advance();
    }

    Program parseProgram() {
        Program prog;
        while (tok_.type != Token::End) {
            if (tok_.type == Token::Newline) { advance(); continue; }
            prog.statements.push_back(parseStatement());
            // optionally consume trailing newlines
            while (tok_.type == Token::Newline) advance();
        }
        return prog;
    }

private:
    TokenStream toks_;
    Token tok_;

    void advance() { tok_ = toks_.consume(); }
    void skipNewlines() { while (tok_.type == Token::Newline) advance(); }

    static bool startsWithUpper(const std::string& s) {
        if (s.empty()) return false;
        return std::isupper(static_cast<unsigned char>(s[0])) != 0;
    }
    static bool startsWithLower(const std::string& s) {
        if (s.empty()) return false;
        return std::islower(static_cast<unsigned char>(s[0])) != 0;
    }

    [[noreturn]] void errorHere(const std::string& msg) const {
        std::ostringstream oss;
        oss << "Parse error at line " << tok_.loc.line << ", col " << tok_.loc.column << ": " << msg;
        throw ParseError(oss.str());
    }

    bool accept(Token::Type t) {
        if (tok_.type == t) { advance(); return true; }
        return false;
    }

    void expect(Token::Type t, const char* what) {
        if (!accept(t)) {
            errorHere(std::string("expected ") + what);
        }
    }

    Identifier parseIdentifier() {
        if (tok_.type != Token::Identifier) errorHere("identifier expected");
        Identifier id{tok_.text, tok_.loc};
        advance();
        return id;
    }

    NumberLiteral parseNumber() {
        if (tok_.type != Token::Integer && tok_.type != Token::Float) errorHere("number expected");
        NumberLiteral n{tok_.text, tok_.loc};
        advance();
        return n;
    }

    StringLiteral parseString() {
        if (tok_.type != Token::String) errorHere("string expected");
        StringLiteral s{tok_.text, tok_.loc};
        advance();
        return s;
    }

    Index parseIndex() {
        Index idx; idx.loc = tok_.loc;
        // Check for virtual index: *identifier [+/-offset] or *integer
        if (tok_.type == Token::Star) {
            SourceLocation loc = tok_.loc;
            advance(); // consume '*'
            Identifier id;
            if (tok_.type == Token::Identifier) {
                id = parseIdentifier();
            } else if (tok_.type == Token::Integer) {
                // Allow *N syntax for queries (e.g., avg[*5] means "at virtual time 5")
                NumberLiteral num = parseNumber();
                id = Identifier{num.text, num.loc};
            } else {
                errorHere("identifier or integer expected after '*' in virtual index");
            }
            int offset = 0;
            // Check for +N or -N offset
            if (tok_.type == Token::Plus) {
                advance();
                if (tok_.type != Token::Integer) errorHere("integer expected after '+' in virtual index");
                NumberLiteral num = parseNumber();
                offset = std::stoi(num.text);
            } else if (tok_.type == Token::Minus) {
                advance();
                if (tok_.type != Token::Integer) errorHere("integer expected after '-' in virtual index");
                NumberLiteral num = parseNumber();
                offset = -std::stoi(num.text);
            }
            VirtualIndex vidx{std::move(id), offset, loc};
            idx.value = std::move(vidx);
            idx.loc = loc;
        } else if (tok_.type == Token::Identifier) {
            auto id = parseIdentifier();
            // Support simple division in index like i/2 used for pooling strides
            if (tok_.type == Token::Slash) {
                advance();
                if (tok_.type != Token::Integer) errorHere("expected integer after '/' in index expression");
                NumberLiteral div = parseNumber();
                // Encode as composite identifier text "i/2" for minimal AST change
                Identifier composed{id.name + "/" + div.text, id.loc};
                id = std::move(composed);
            }
            // optional normalization dot suffix: i.
            if (tok_.type == Token::Dot) {
                advance(); // ignore normalization for now
            }
            idx.value = std::move(id);
        } else if (tok_.type == Token::Integer) {
            idx.value = parseNumber();
        } else {
            errorHere("index (identifier, integer, or virtual index) expected");
        }
        return idx;
    }

    std::vector<Index> parseIndexList() {
        std::vector<Index> v;
        v.push_back(parseIndex());
        while (accept(Token::Comma)) {
            v.push_back(parseIndex());
        }
        return v;
    }

    TensorRef parseTensorRef() {
        TensorRef ref; ref.loc = tok_.loc;
        ref.name = parseIdentifier();
        if (accept(Token::LBracket)) {
            if (tok_.type != Token::RBracket) {
                ref.indices = parseIndexList();
            }
            expect(Token::RBracket, "]");
        }
        return ref;
    }

    // Expressions: term ((+|-) term)*, where term is product of primaries by implicit multiplication
    ExprPtr parseExpr() {
        skipNewlines();
        auto lhs = parseTerm();
        while (true) {
            skipNewlines();
            if (tok_.type == Token::Plus || tok_.type == Token::Minus) {
                Token::Type op = tok_.type; advance();
                skipNewlines();
                auto rhs = parseTerm();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin;
                bin.op = (op == Token::Plus) ? ExprBinary::Op::Add : ExprBinary::Op::Sub;
                bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            break;
        }
        return lhs;
    }

    // Parse a primary: number | tensor_ref | '(' expr ')' | list literal [...] | unary '-' primary
    ExprPtr parsePrimary() {
        // unary minus
        if (tok_.type == Token::Minus) {
            SourceLocation loc = tok_.loc;
            advance();
            auto rhs = parsePrimary();
            // build 0 - rhs
            NumberLiteral zero{"0", loc};
            auto zeroExpr = std::make_shared<Expr>(); zeroExpr->loc = loc; zeroExpr->node = ExprNumber{zero};
            auto e = std::make_shared<Expr>(); e->loc = loc; ExprBinary bin; bin.op = ExprBinary::Op::Sub; bin.lhs = zeroExpr; bin.rhs = rhs; e->node = std::move(bin);
            return e;
        }
        if (tok_.type == Token::LParen) {
            advance();
            auto inner = parseExpr();
            expect(Token::RParen, ")");
            auto e = std::make_shared<Expr>(); e->loc = inner->loc; e->node = ExprParen{inner};
            return e;
        }
        if (tok_.type == Token::LBracket) {
            // Nested list literal: elements are expressions or sublists
            SourceLocation loc = tok_.loc;
            advance(); // consume '['
            std::vector<ExprPtr> elems;
            if (tok_.type != Token::RBracket) {
                elems.push_back(parseExpr());
                while (accept(Token::Comma)) {
                    elems.push_back(parseExpr());
                }
            }
            expect(Token::RBracket, "]");
            auto e = std::make_shared<Expr>(); e->loc = loc; e->node = ExprList{std::move(elems)};
            return e;
        }
        if (tok_.type == Token::Integer || tok_.type == Token::Float) {
            auto num = parseNumber();
            auto e = std::make_shared<Expr>(); e->loc = num.loc; e->node = ExprNumber{num};
            return e;
        }
        // identifier: could be function call or tensor ref
        if (tok_.type == Token::Identifier) {
            // Lookahead for '('
            Identifier id = parseIdentifier();
            if (tok_.type == Token::LParen) {
                // function call: id '(' args ')'
                SourceLocation loc = id.loc;
                advance(); // consume '('
                std::vector<ExprPtr> args;
                if (tok_.type != Token::RParen) {
                    // parse first expr
                    args.push_back(parseExpr());
                    while (accept(Token::Comma)) {
                        args.push_back(parseExpr());
                    }
                }
                expect(Token::RParen, ")");
                auto e = std::make_shared<Expr>(); e->loc = loc; e->node = ExprCall{std::move(id), std::move(args)};
                return e;
            } else {
                // tensor ref or scalar identifier (no indices)
                TensorRef ref; ref.loc = id.loc; ref.name = std::move(id);
                if (accept(Token::LBracket)) {
                    if (tok_.type != Token::RBracket) {
                        ref.indices = parseIndexList();
                    }
                    expect(Token::RBracket, "]");
                }
                auto e = std::make_shared<Expr>(); e->loc = ref.loc; e->node = ExprTensorRef{ref};
                return e;
            }
        }
        // Optional: allow bare strings as primaries for now (kept for compatibility)
        if (tok_.type == Token::String) {
            auto s = parseString();
            auto e = std::make_shared<Expr>(); e->loc = s.loc; e->node = ExprString{s};
            return e;
        }
        errorHere("expression expected");
        // Unreachable
        // return {};
    }

    // term := primary { (('*' | '/' | '%') primary) | primary }  // support explicit division, modulo, and implicit multiplication
    ExprPtr parseTerm() {
        auto lhs = parsePrimary();
        auto startsPrimary = [&](Token::Type t)->bool {
            return t == Token::Identifier || t == Token::Integer || t == Token::Float || t == Token::LParen; // exclude String from implicit mul
        };
        while (true) {
            if (tok_.type == Token::Slash) {
                advance();
                auto rhs = parsePrimary();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin; bin.op = ExprBinary::Op::Div; bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            if (tok_.type == Token::Star) {
                advance();
                auto rhs = parsePrimary();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin; bin.op = ExprBinary::Op::Mul; bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            if (tok_.type == Token::Percent) {
                advance();
                auto rhs = parsePrimary();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin; bin.op = ExprBinary::Op::Mod; bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            if (startsPrimary(tok_.type)) {
                auto rhs = parsePrimary();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin; bin.op = ExprBinary::Op::Mul; bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            break;
        }
        return lhs;
    }

    // Guard condition parsing for guarded clauses
    // parseGuardComparison: handles comparisons of arithmetic expressions
    ExprPtr parseGuardComparison() {
        auto lhs = parseExpr();
        // Check for comparison operators
        if (tok_.type == Token::Less || tok_.type == Token::Le ||
            tok_.type == Token::Greater || tok_.type == Token::Ge ||
            tok_.type == Token::EqEq || tok_.type == Token::NotEq) {
            Token::Type opType = tok_.type;
            advance();
            auto rhs = parseExpr();
            auto e = std::make_shared<Expr>();
            e->loc = lhs->loc;
            ExprBinary bin;
            switch (opType) {
                case Token::Less: bin.op = ExprBinary::Op::Lt; break;
                case Token::Le: bin.op = ExprBinary::Op::Le; break;
                case Token::Greater: bin.op = ExprBinary::Op::Gt; break;
                case Token::Ge: bin.op = ExprBinary::Op::Ge; break;
                case Token::EqEq: bin.op = ExprBinary::Op::Eq; break;
                case Token::NotEq: bin.op = ExprBinary::Op::Ne; break;
                default: errorHere("internal error: unexpected comparison operator");
            }
            bin.lhs = lhs;
            bin.rhs = rhs;
            e->node = std::move(bin);
            return e;
        }
        // If no comparison operator, return the expression as-is (for boolean values)
        return lhs;
    }

    // parseGuardNotFactor: handles 'not' prefix operator and parenthesized conditions
    ExprPtr parseGuardNotFactor() {
        if (tok_.type == Token::KwNot) {
            SourceLocation loc = tok_.loc;
            advance();
            auto operand = parseGuardNotFactor();
            auto e = std::make_shared<Expr>();
            e->loc = loc;
            ExprUnary un;
            un.op = ExprUnary::Op::Not;
            un.operand = operand;
            e->node = std::move(un);
            return e;
        }
        if (tok_.type == Token::LParen) {
            advance();
            auto inner = parseGuardCondition();
            expect(Token::RParen, ")");
            return inner;
        }
        return parseGuardComparison();
    }

    // parseGuardAndTerm: handles 'and' operator
    ExprPtr parseGuardAndTerm() {
        auto lhs = parseGuardNotFactor();
        while (tok_.type == Token::KwAnd) {
            advance();
            auto rhs = parseGuardNotFactor();
            auto e = std::make_shared<Expr>();
            e->loc = lhs->loc;
            ExprBinary bin;
            bin.op = ExprBinary::Op::And;
            bin.lhs = lhs;
            bin.rhs = rhs;
            e->node = std::move(bin);
            lhs = e;
        }
        return lhs;
    }

    // parseGuardCondition: handles 'or' operator (lowest precedence)
    ExprPtr parseGuardCondition() {
        auto lhs = parseGuardAndTerm();
        while (tok_.type == Token::KwOr) {
            advance();
            auto rhs = parseGuardAndTerm();
            auto e = std::make_shared<Expr>();
            e->loc = lhs->loc;
            ExprBinary bin;
            bin.op = ExprBinary::Op::Or;
            bin.lhs = lhs;
            bin.rhs = rhs;
            e->node = std::move(bin);
            lhs = e;
        }
        return lhs;
    }

    // parseGuardedClause: parses expression [: guard_condition]
    GuardedClause parseGuardedClause() {
        skipNewlines();
        GuardedClause clause;
        clause.expr = parseExpr();
        clause.loc = clause.expr->loc;
        if (accept(Token::Colon)) {
            skipNewlines();
            clause.guard = parseGuardCondition();
        }
        return clause;
    }

    // Datalog parsing
    Identifier parseLowercaseIdentifier() {
        if (tok_.type != Token::Identifier || !startsWithLower(tok_.text)) errorHere("lowercase identifier expected");
        return parseIdentifier();
    }

    std::variant<Identifier, StringLiteral> parseDatalogTerm() {
        if (tok_.type == Token::String || tok_.type == Token::Integer || (tok_.type == Token::Identifier && startsWithUpper(tok_.text))) {
            // constant
            return parseDatalogConstant();
        }
        if (tok_.type == Token::Identifier && startsWithLower(tok_.text)) {
            // variable
            return parseLowercaseIdentifier();
        }
        errorHere("datalog term expected (variable or constant)");
        // Unreachable
        // return Identifier{};
    }

    std::vector<std::variant<Identifier, StringLiteral>> parseDatalogTermList() {
        std::vector<std::variant<Identifier, StringLiteral>> v;
        v.push_back(parseDatalogTerm());
        while (accept(Token::Comma)) v.push_back(parseDatalogTerm());
        return v;
    }

    DatalogAtom parseAtom() {
        if (tok_.type != Token::Identifier || !startsWithUpper(tok_.text)) errorHere("relation (Uppercase Identifier) expected");
        Identifier rel = parseIdentifier();
        SourceLocation loc = rel.loc;
        expect(Token::LParen, "(");
        std::vector<std::variant<Identifier, StringLiteral>> terms;
        if (tok_.type != Token::RParen) {
            terms = parseDatalogTermList();
        }
        expect(Token::RParen, ")");
        DatalogAtom a; a.relation = std::move(rel); a.terms = std::move(terms); a.loc = loc; return a;
    }

    static bool allConstants(const DatalogAtom& a) {
        return std::all_of(a.terms.begin(), a.terms.end(),
                           [](const auto &t) { return !std::holds_alternative<Identifier>(t); });
    }

    // Comparison operator acceptance: fills opOut with textual op and consumes token
    bool acceptComparison(std::string& opOut) {
        switch (tok_.type) {
            case Token::Ge: opOut = ">="; advance(); return true;
            case Token::Le: opOut = "<="; advance(); return true;
            case Token::EqEq: opOut = "=="; advance(); return true;
            case Token::NotEq: opOut = "!="; advance(); return true;
            case Token::Greater: opOut = ">"; advance(); return true;
            case Token::Less: opOut = "<"; advance(); return true;
            default: return false;
        }
    }

    DatalogCondition parseComparisonCondition() {
        auto lhs = parseExpr();
        std::string op;
        if (!acceptComparison(op)) {
            errorHere("comparison operator expected (>, <, >=, <=, ==, !=)");
        }
        auto rhs = parseExpr();
        DatalogCondition c; c.lhs = lhs; c.op = op; c.rhs = rhs; c.loc = lhs->loc; return c;
    }

    std::variant<DatalogAtom, DatalogCondition> parseRuleBodyElement() {
        // Allow body elements to span lines
        skipNewlines();
        // Decide based on lookahead: Uppercase Identifier followed by '(' means atom
        if (tok_.type == Token::Identifier && startsWithUpper(tok_.text) && toks_.peek().type == Token::LParen) {
            return parseAtom();
        }
        return parseComparisonCondition();
    }

    // Parse a Datalog constant minimally into a StringLiteral (uppercase identifiers or numbers or strings)
    StringLiteral parseDatalogConstant() {
        if (tok_.type == Token::String) {
            return parseString();
        }
        if (tok_.type == Token::Integer) {
            NumberLiteral n = parseNumber();
            return StringLiteral{n.text, n.loc};
        }
        if (tok_.type == Token::Identifier && startsWithUpper(tok_.text)) {
            const Identifier id = parseIdentifier();
            return StringLiteral{id.name, id.loc};
        }
        errorHere("datalog constant expected (String, Integer, or Uppercase Identifier)");
        // Unreachable
        // return {};
    }

    static DatalogFact convertAtomToFact(const DatalogAtom& a) {
        DatalogFact f; f.relation = a.relation; f.loc = a.loc;
        for (const auto& t : a.terms) {
            f.constants.push_back(std::get<StringLiteral>(t));
        }
        return f;
    }

    Statement parseStatement() {
        // Possible Datalog atom/rule/fact at statement start
        if (tok_.type == Token::Identifier && tok_.text != "file" && toks_.peek().type == Token::LParen && startsWithUpper(tok_.text)) {
            DatalogAtom head = parseAtom();
            // Datalog query: Atom? or Atom, body...? (conjunctive query with comparisons)
            if (accept(Token::Question)) {
                Query q; q.target = head; q.loc = head.loc; return q;
            }
            if (accept(Token::LArrow)) {
                // parse body: elements are either Atom or Condition (comparison of tensor expressions)
                std::vector<std::variant<DatalogAtom, DatalogCondition>> body;
                body.push_back(parseRuleBodyElement());
                while (accept(Token::Comma)) body.push_back(parseRuleBodyElement());
                DatalogRule r; r.head = std::move(head); r.body = std::move(body); r.loc = r.head.loc;
                return r;
            }
            // Datalog conjunctive query: Atom, (Atom|Condition) ... ?
            if (tok_.type == Token::Comma) {
                std::vector<std::variant<DatalogAtom, DatalogCondition>> conj;
                // include the first atom
                conj.push_back(head);
                do {
                    advance(); // consume ','
                    conj.push_back(parseRuleBodyElement());
                } while (tok_.type == Token::Comma);
                expect(Token::Question, "'?' to end query");
                Query q; q.target = head; q.body = std::move(conj); q.loc = head.loc; return q;
            }
            // else, treat as fact if constants only
            if (allConstants(head)) {
                return convertAtomToFact(head);
            } else {
                errorHere("expected '<-' to form a rule or constants-only fact or '?' for query");
            }
        }
        // File operation could start with 'file' or string literal; we only implement string variant and normalize
        if (tok_.type == Token::Identifier && tok_.text == "file") {
            // file("path") = TensorRef
            auto fileLit = parseFileLiteral();
            expect(Token::Equals, "=");
            auto tr = parseTensorRef();
            FileOperation fo; fo.lhsIsTensor = false; fo.tensor = tr; fo.file = fileLit; fo.loc = fileLit.loc;
            return fo;
        }
        if (tok_.type == Token::String) {
            // "path" = TensorRef
            auto s = parseString();
            expect(Token::Equals, "=");
            auto tr = parseTensorRef();
            FileOperation fo; fo.lhsIsTensor = false; fo.tensor = tr; fo.file = s; fo.loc = s.loc;
            return fo;
        }
        // Query: tensor_ref?
        {
            // Try to parse tensor_ref followed by '?'
            auto save = tok_;
            try {
                auto tr = parseTensorRef();
                if (accept(Token::Question)) {
                    Query q; q.target = tr; q.loc = tr.loc; return q;
                }
                // Not a query; fallthrough to equation
                // Reset not possible easily; instead, if no '?', we consider it as LHS already consumed
                // Build equation from existing LHS
                // Parse projection operator: '=', '+=', 'avg=', 'max=', 'min='
                std::string proj = "=";
                if (tok_.type == Token::Plus) {
                    advance(); expect(Token::Equals, "="); proj = "+=";
                } else if (tok_.type == Token::Identifier && (tok_.text == "avg" || tok_.text == "max" || tok_.text == "min")) {
                    std::string op = tok_.text; advance(); expect(Token::Equals, "="); proj = op + "=";
                } else {
                    expect(Token::Equals, "projection '='");
                }
                // Support file operation: tensor_ref = file_literal
                if ((tok_.type == Token::Identifier && tok_.text == "file") || tok_.type == Token::String) {
                    StringLiteral s = (tok_.type == Token::String) ? parseString() : parseFileLiteral();
                    FileOperation fo; fo.lhsIsTensor = true; fo.tensor = tr; fo.file = s; fo.loc = s.loc;
                    return fo;
                }
                // Parse guarded clauses separated by '|'
                TensorEquation eq; eq.lhs = tr; eq.projection = proj; eq.loc = tr.loc;
                eq.clauses.push_back(parseGuardedClause());
                skipNewlines(); // Allow newlines before '|'
                while (accept(Token::Pipe)) {
                    eq.clauses.push_back(parseGuardedClause());
                    skipNewlines(); // Allow newlines before next '|'
                }
                return eq;
            } catch (const ParseError&) {
                // If failed, restore token best-effort (no backtracking of token stream provided)
                tok_ = save;
            }
        }
        // Fallback explicit equation parse
        TensorRef lhs = parseTensorRef();
        // Parse projection operator: '=', '+=', 'avg=', 'max=', 'min='
        std::string proj = "=";
        if (tok_.type == Token::Plus) {
            advance(); expect(Token::Equals, "="); proj = "+=";
        } else if (tok_.type == Token::Identifier && (tok_.text == "avg" || tok_.text == "max" || tok_.text == "min")) {
            std::string op = tok_.text; advance(); expect(Token::Equals, "="); proj = op + "=";
        } else {
            expect(Token::Equals, "projection '='");
        }
        // Support file operation: tensor_ref = file_literal (fallback path)
        if ((tok_.type == Token::Identifier && tok_.text == "file") || tok_.type == Token::String) {
            StringLiteral s = (tok_.type == Token::String) ? parseString() : parseFileLiteral();
            FileOperation fo; fo.lhsIsTensor = true; fo.tensor = lhs; fo.file = s; fo.loc = s.loc;
            return fo;
        }
        // Parse guarded clauses separated by '|'
        TensorEquation eq; eq.lhs = lhs; eq.projection = proj; eq.loc = lhs.loc;
        eq.clauses.push_back(parseGuardedClause());
        skipNewlines(); // Allow newlines before '|'
        while (accept(Token::Pipe)) {
            eq.clauses.push_back(parseGuardedClause());
            skipNewlines(); // Allow newlines before next '|'
        }
        return eq;
    }

    StringLiteral parseFileLiteral() {
        // file("...") or raw string already handled elsewhere
        auto id = parseIdentifier();
        if (id.name != "file") errorHere("expected file(");
        expect(Token::LParen, "(");
        auto s = parseString();
        expect(Token::RParen, ")");
        return s;
    }
};

} // namespace

Program parseProgram(const std::string_view source) {
    Parser p(source);
    return p.parseProgram();
}

Program parseFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw ParseError("Cannot open file: " + path);
    std::stringstream buffer; buffer << ifs.rdbuf();
    return parseProgram(buffer.str());
}

} // namespace tl

