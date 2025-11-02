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
            idx.value = std::move(id);
            // Check for normalization dot suffix: i.
            if (tok_.type == Token::Dot) {
                advance();
                idx.normalized = true;
            }
        } else if (tok_.type == Token::Integer) {
            idx.value = parseNumber();
        } else {
            errorHere("index (identifier, integer, or virtual index) expected");
        }
        return idx;
    }

    // Parse a slice: start:end:step, :end, start:, :, ::step, start::step, or just an integer/identifier
    Slice parseSlice() {
        Slice slice;
        slice.loc = tok_.loc;

        // Check for leading colon (: or :end or ::step forms)
        if (tok_.type == Token::Colon) {
            advance(); // consume first ':'
            // Check if there's another colon immediately (::step form)
            if (tok_.type == Token::Colon) {
                advance(); // consume second ':'
                if (tok_.type != Token::Integer) errorHere("integer expected for step in slice");
                slice.step = parseNumber();
                return slice;
            }
            // Check if there's an end value (:end or :end:step forms)
            if (tok_.type == Token::Integer) {
                slice.end = parseNumber();
                // Check for step (:end:step)
                if (tok_.type == Token::Colon) {
                    advance(); // consume second ':'
                    if (tok_.type != Token::Integer) errorHere("integer expected for step in slice");
                    slice.step = parseNumber();
                }
            }
            // else it's just ':', a full slice
            return slice;
        }

        // Must start with integer (start:... or start)
        if (tok_.type != Token::Integer) {
            errorHere("slice must start with integer or ':'");
        }

        slice.start = parseNumber();

        // Check for colon to continue slice (start: or start:end or start:end:step or start::step)
        if (tok_.type == Token::Colon) {
            advance(); // consume first ':'
            // Check if there's another colon immediately (start::step form)
            if (tok_.type == Token::Colon) {
                advance(); // consume second ':'
                if (tok_.type != Token::Integer) errorHere("integer expected for step in slice");
                slice.step = parseNumber();
                return slice;
            }
            // Check if there's an end value (start:end or start:end:step forms)
            if (tok_.type == Token::Integer) {
                slice.end = parseNumber();
                // Check for step (start:end:step)
                if (tok_.type == Token::Colon) {
                    advance(); // consume second ':'
                    if (tok_.type != Token::Integer) errorHere("integer expected for step in slice");
                    slice.step = parseNumber();
                }
            }
            // else it's start:, slice to end
        }

        return slice;
    }

    // Parse either an index or a slice
    IndexOrSlice parseIndexOrSlice() {
        IndexOrSlice ios;
        ios.loc = tok_.loc;

        // Lookahead: if we see a colon immediately, it's a slice
        if (tok_.type == Token::Colon) {
            ios.value = parseSlice();
            return ios;
        }

        // If we see an integer followed by a colon, it's a slice
        if (tok_.type == Token::Integer && toks_.peek().type == Token::Colon) {
            ios.value = parseSlice();
            return ios;
        }

        // Check for minus (negative slice bound)
        if (tok_.type == Token::Minus && toks_.peek().type == Token::Integer) {
            // This is a negative number, parse as slice
            SourceLocation loc = tok_.loc;
            advance(); // consume '-'
            NumberLiteral num = parseNumber();
            num.text = "-" + num.text; // Make it negative
            num.loc = loc;

            Slice slice;
            slice.loc = loc;
            slice.start = num;

            // Check for colon continuation
            if (tok_.type == Token::Colon) {
                advance(); // consume ':'
                if (tok_.type == Token::Integer || tok_.type == Token::Minus) {
                    if (tok_.type == Token::Minus) {
                        advance();
                        NumberLiteral endNum = parseNumber();
                        endNum.text = "-" + endNum.text;
                        slice.end = endNum;
                    } else {
                        slice.end = parseNumber();
                    }
                }
            }
            ios.value = slice;
            return ios;
        }

        // Otherwise, try to parse as a regular index
        ios.value = parseIndex();
        return ios;
    }

    std::vector<IndexOrSlice> parseIndexOrSliceList() {
        std::vector<IndexOrSlice> v;
        v.push_back(parseIndexOrSlice());
        while (accept(Token::Comma)) {
            v.push_back(parseIndexOrSlice());
        }
        return v;
    }

    TensorRef parseTensorRef() {
        TensorRef ref; ref.loc = tok_.loc;
        ref.name = parseIdentifier();
        if (accept(Token::LBracket)) {
            if (tok_.type != Token::RBracket) {
                ref.indices = parseIndexOrSliceList();
            }
            expect(Token::RBracket, "]");
        }
        return ref;
    }

    // Expressions: comparison (comparison_op comparison)?, with proper precedence
    // Hierarchy: parseExpr -> parseComparison -> parseAddSub -> parseTerm -> parsePower -> parsePrimary
    ExprPtr parseExpr() {
        return parseComparison();
    }

    // parseComparison: handles comparison operators (<, >, <=, >=, ==, !=)
    // These have lower precedence than arithmetic operators
    ExprPtr parseComparison() {
        skipNewlines();
        auto lhs = parseAddSub();
        // Check for comparison operators
        if (tok_.type == Token::Less || tok_.type == Token::Le ||
            tok_.type == Token::Greater || tok_.type == Token::Ge ||
            tok_.type == Token::EqEq || tok_.type == Token::NotEq) {
            Token::Type opType = tok_.type;
            advance();
            skipNewlines();
            auto rhs = parseAddSub();
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
        return lhs;
    }

    // parseAddSub: handles addition and subtraction (higher precedence than comparisons)
    ExprPtr parseAddSub() {
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
                        ref.indices = parseIndexOrSliceList();
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

    // power := primary [ '^' power ]  // right-associative exponentiation
    ExprPtr parsePower() {
        auto lhs = parsePrimary();
        if (tok_.type == Token::Caret) {
            advance();
            auto rhs = parsePower();  // right-associative: recurse for right side
            auto e = std::make_shared<Expr>();
            e->loc = lhs->loc;
            ExprBinary bin; bin.op = ExprBinary::Op::Pow; bin.lhs = lhs; bin.rhs = rhs;
            e->node = std::move(bin);
            return e;
        }
        return lhs;
    }

    // term := power { (('*' | '/' | '%') power) | power }  // support explicit division, modulo, and implicit multiplication
    ExprPtr parseTerm() {
        auto lhs = parsePower();
        auto startsPrimary = [&](Token::Type t)->bool {
            return t == Token::Identifier || t == Token::Integer || t == Token::Float || t == Token::LParen; // exclude String from implicit mul
        };
        while (true) {
            if (tok_.type == Token::Slash) {
                advance();
                auto rhs = parsePower();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin; bin.op = ExprBinary::Op::Div; bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            if (tok_.type == Token::Star) {
                advance();
                auto rhs = parsePower();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin; bin.op = ExprBinary::Op::Mul; bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            if (tok_.type == Token::Percent) {
                advance();
                auto rhs = parsePower();
                auto e = std::make_shared<Expr>();
                e->loc = lhs->loc;
                ExprBinary bin; bin.op = ExprBinary::Op::Mod; bin.lhs = lhs; bin.rhs = rhs;
                e->node = std::move(bin);
                lhs = e;
                continue;
            }
            if (startsPrimary(tok_.type)) {
                auto rhs = parsePower();
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
    // Now uses parseAddSub instead of parseExpr to maintain proper precedence
    ExprPtr parseGuardComparison() {
        auto lhs = parseAddSub();
        // Check for comparison operators
        if (tok_.type == Token::Less || tok_.type == Token::Le ||
            tok_.type == Token::Greater || tok_.type == Token::Ge ||
            tok_.type == Token::EqEq || tok_.type == Token::NotEq) {
            Token::Type opType = tok_.type;
            advance();
            auto rhs = parseAddSub();
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

    std::variant<Identifier, StringLiteral, ExprPtr> parseDatalogTerm() {
        // Check for arithmetic expressions (binary operators indicate complex expression)
        // Parse primary term first
        if (tok_.type == Token::String) {
            auto constant = parseDatalogConstant();
            // parseDatalogConstant returns variant<StringLiteral, NumberLiteral>
            // For strings, it returns StringLiteral
            return std::get<StringLiteral>(constant);
        }
        if (tok_.type == Token::Integer || tok_.type == Token::Float) {
            // Check if this is part of an arithmetic expression
            auto num = parseNumber();
            // Check for arithmetic operators
            if (tok_.type == Token::Plus || tok_.type == Token::Minus ||
                tok_.type == Token::Star || tok_.type == Token::Slash ||
                tok_.type == Token::Percent) {
                // This is an arithmetic expression, parse it fully
                auto expr = std::make_shared<Expr>();
                expr->loc = num.loc;
                expr->node = ExprNumber{num};
                // Now parse the rest of the expression
                expr = parseDatalogArithmeticFrom(expr);
                return expr;
            }
            // Just a constant number - return as StringLiteral for now
            return StringLiteral{num.text, num.loc};
        }
        if (tok_.type == Token::Identifier) {
            if (startsWithUpper(tok_.text)) {
                // constant (uppercase identifier)
                auto constant = parseDatalogConstant();
                // For uppercase identifiers, it returns StringLiteral
                return std::get<StringLiteral>(constant);
            } else {
                // variable (lowercase identifier) or part of arithmetic expression
                auto id = parseLowercaseIdentifier();
                // Check for arithmetic operators
                if (tok_.type == Token::Plus || tok_.type == Token::Minus ||
                    tok_.type == Token::Star || tok_.type == Token::Slash ||
                    tok_.type == Token::Percent) {
                    // This is an arithmetic expression
                    auto expr = std::make_shared<Expr>();
                    expr->loc = id.loc;
                    TensorRef ref;
                    ref.name = id;
                    ref.loc = id.loc;
                    expr->node = ExprTensorRef{ref};
                    // Parse the rest of the expression
                    expr = parseDatalogArithmeticFrom(expr);
                    return expr;
                }
                // Just a variable
                return id;
            }
        }
        if (tok_.type == Token::LParen) {
            // Parenthesized expression - use parseExpr for full expression support
            advance();
            auto inner = parseExpr();
            expect(Token::RParen, ")");
            // Check if there's a continuation with arithmetic operators
            if (tok_.type == Token::Plus || tok_.type == Token::Minus ||
                tok_.type == Token::Star || tok_.type == Token::Slash ||
                tok_.type == Token::Percent) {
                // Continue parsing the arithmetic expression
                inner = parseDatalogArithmeticFrom(inner);
            }
            return inner;
        }
        errorHere("datalog term expected (variable, constant, or arithmetic expression)");
        // Unreachable
        // return Identifier{};
    }

    // Helper to continue parsing arithmetic expression from a left-hand side with proper precedence
    // This handles addition/subtraction (lower precedence)
    ExprPtr parseDatalogArithmeticFrom(ExprPtr lhs) {
        // First, handle any pending multiplication/division (higher precedence)
        lhs = parseDatalogArithmeticMulDiv(lhs);

        // Then handle addition/subtraction
        while (tok_.type == Token::Plus || tok_.type == Token::Minus) {
            Token::Type op = tok_.type;
            advance();
            ExprPtr rhs = parseDatalogArithmeticMulDiv(parseDatalogArithmeticPrimary());
            auto e = std::make_shared<Expr>();
            e->loc = lhs->loc;
            ExprBinary bin;
            bin.op = (op == Token::Plus) ? ExprBinary::Op::Add : ExprBinary::Op::Sub;
            bin.lhs = lhs;
            bin.rhs = rhs;
            e->node = std::move(bin);
            lhs = e;
        }
        return lhs;
    }

    // Helper to parse multiplication/division/modulo (higher precedence than add/sub)
    ExprPtr parseDatalogArithmeticMulDiv(ExprPtr lhs) {
        while (tok_.type == Token::Star || tok_.type == Token::Slash || tok_.type == Token::Percent) {
            Token::Type op = tok_.type;
            advance();
            ExprPtr rhs = parseDatalogArithmeticPrimary();
            auto e = std::make_shared<Expr>();
            e->loc = lhs->loc;
            ExprBinary bin;
            if (op == Token::Star) bin.op = ExprBinary::Op::Mul;
            else if (op == Token::Slash) bin.op = ExprBinary::Op::Div;
            else bin.op = ExprBinary::Op::Mod;
            bin.lhs = lhs;
            bin.rhs = rhs;
            e->node = std::move(bin);
            lhs = e;
        }
        return lhs;
    }

    // Parse a primary in datalog arithmetic context
    ExprPtr parseDatalogArithmeticPrimary() {
        if (tok_.type == Token::Integer) {
            auto num = parseNumber();
            auto e = std::make_shared<Expr>();
            e->loc = num.loc;
            e->node = ExprNumber{num};
            return e;
        }
        if (tok_.type == Token::Identifier && startsWithLower(tok_.text)) {
            auto id = parseLowercaseIdentifier();
            auto e = std::make_shared<Expr>();
            e->loc = id.loc;
            TensorRef ref;
            ref.name = id;
            ref.loc = id.loc;
            e->node = ExprTensorRef{ref};
            return e;
        }
        if (tok_.type == Token::LParen) {
            advance();
            auto inner = parseExpr();
            expect(Token::RParen, ")");
            return inner;
        }
        errorHere("arithmetic primary expected (number, variable, or parenthesized expression)");
        // Unreachable
        // return {};
    }

    std::vector<std::variant<Identifier, StringLiteral, ExprPtr>> parseDatalogTermList() {
        std::vector<std::variant<Identifier, StringLiteral, ExprPtr>> v;
        v.push_back(parseDatalogTerm());
        while (accept(Token::Comma)) v.push_back(parseDatalogTerm());
        return v;
    }

    DatalogAtom parseAtom() {
        if (tok_.type != Token::Identifier || !startsWithUpper(tok_.text)) errorHere("relation (Uppercase Identifier) expected");
        Identifier rel = parseIdentifier();
        SourceLocation loc = rel.loc;
        expect(Token::LParen, "(");
        std::vector<std::variant<Identifier, StringLiteral, ExprPtr>> terms;
        if (tok_.type != Token::RParen) {
            terms = parseDatalogTermList();
        }
        expect(Token::RParen, ")");
        DatalogAtom a; a.relation = std::move(rel); a.terms = std::move(terms); a.loc = loc; return a;
    }

    static bool allConstants(const DatalogAtom& a) {
        return std::all_of(a.terms.begin(), a.terms.end(),
                           [](const auto &t) {
                               // Constants are StringLiterals (not Identifiers, not general ExprPtr)
                               // But we also accept simple number literals (ExprPtr containing only ExprNumber)
                               if (std::holds_alternative<StringLiteral>(t)) {
                                   return true;
                               }
                               // Check if it's an expression containing just a number
                               if (auto* expr = std::get_if<ExprPtr>(&t)) {
                                   return std::holds_alternative<ExprNumber>((*expr)->node);
                               }
                               return false;
                           });
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
        auto lhs = parseAddSub();  // Use parseAddSub to avoid including comparison in subexpressions
        std::string op;
        if (!acceptComparison(op)) {
            errorHere("comparison operator expected (>, <, >=, <=, ==, !=)");
        }
        auto rhs = parseAddSub();
        DatalogCondition c; c.lhs = lhs; c.op = op; c.rhs = rhs; c.loc = lhs->loc; return c;
    }

    std::variant<DatalogAtom, DatalogNegation, DatalogCondition> parseRuleBodyElement() {
        // Allow body elements to span lines
        skipNewlines();
        // Handle negation keyword before an atom
        if (tok_.type == Token::KwNot) {
            advance();
            skipNewlines();
            DatalogNegation neg;
            neg.atom = parseAtom();
            neg.loc = neg.atom.loc;
            return neg;
        }
        // Decide based on lookahead: Uppercase Identifier followed by '(' means atom
        if (tok_.type == Token::Identifier && startsWithUpper(tok_.text) && toks_.peek().type == Token::LParen) {
            return parseAtom();
        }
        return parseComparisonCondition();
    }

    // Parse a directive argument: name=value
    DirectiveArg parseDirectiveArg() {
        DirectiveArg arg;
        arg.name = parseIdentifier();
        arg.loc = arg.name.loc;
        expect(Token::Equals, "= in directive argument");

        // Value can be a number, string, or boolean (true/false keywords)
        if (tok_.type == Token::Integer || tok_.type == Token::Float) {
            arg.value = parseNumber();
        } else if (tok_.type == Token::String) {
            arg.value = parseString();
        } else if (tok_.type == Token::Identifier) {
            std::string val = tok_.text;
            if (val == "true" || val == "True") {
                advance();
                arg.value = true;
            } else if (val == "false" || val == "False") {
                advance();
                arg.value = false;
            } else {
                errorHere("expected number, string, or boolean (true/false) for directive argument value");
            }
        } else {
            errorHere("expected number, string, or boolean for directive argument value");
        }
        return arg;
    }

    // Parse a query directive: @directiveName(arg1=val1, arg2=val2, ...)
    std::optional<QueryDirective> parseQueryDirective() {
        if (!accept(Token::At)) {
            return std::nullopt;
        }

        QueryDirective dir;
        dir.loc = tok_.loc;
        dir.name = parseIdentifier();
        expect(Token::LParen, "( after directive name");

        // Parse arguments
        if (tok_.type != Token::RParen) {
            dir.args.push_back(parseDirectiveArg());
            while (accept(Token::Comma)) {
                dir.args.push_back(parseDirectiveArg());
            }
        }
        expect(Token::RParen, ") to close directive");

        return dir;
    }

    // Parse a Datalog constant into a StringLiteral or NumberLiteral (uppercase identifiers, numbers, or strings)
    std::variant<StringLiteral, NumberLiteral> parseDatalogConstant() {
        if (tok_.type == Token::String) {
            return parseString();
        }
        if (tok_.type == Token::Integer || tok_.type == Token::Float) {
            return parseNumber();
        }
        if (tok_.type == Token::Identifier && startsWithUpper(tok_.text)) {
            const Identifier id = parseIdentifier();
            return StringLiteral{id.name, id.loc};
        }
        errorHere("datalog constant expected (String, Number, or Uppercase Identifier)");
        // Unreachable
        // return {};
    }

    static DatalogFact convertAtomToFact(const DatalogAtom& a) {
        DatalogFact f; f.relation = a.relation; f.loc = a.loc;
        for (const auto& t : a.terms) {
            if (auto* sl = std::get_if<StringLiteral>(&t)) {
                f.constants.push_back(*sl);
            } else if (auto* id = std::get_if<Identifier>(&t)) {
                // If the term is an Identifier (shouldn't happen for facts, but handle it)
                f.constants.push_back(StringLiteral{id->name, id->loc});
            } else if (auto* expr = std::get_if<ExprPtr>(&t)) {
                // If the term is an expression, try to extract number literal
                if (auto* exprNum = std::get_if<ExprNumber>(&(*expr)->node)) {
                    f.constants.push_back(exprNum->literal);
                } else {
                    throw ParseError("Datalog facts can only contain constants (not expressions)");
                }
            }
        }
        return f;
    }

    // Validate normalized indices constraints
    void validateNormalizedIndices(const TensorEquation& eq) {
        int normalizedCount = 0;
        const Index* normalizedIndex = nullptr;

        // Check LHS for normalized indices
        for (const auto& ios : eq.lhs.indices) {
            if (auto* idx = std::get_if<Index>(&ios.value)) {
                if (idx->normalized) {
                    normalizedCount++;
                    normalizedIndex = idx;

                    // Constraint: normalized index must be a free variable (lowercase identifier)
                    if (auto* ident = std::get_if<Identifier>(&idx->value)) {
                        if (!startsWithLower(ident->name)) {
                            std::ostringstream oss;
                            oss << "Normalized index must be a free variable (lowercase identifier), got '"
                                << ident->name << "'";
                            throw ParseError(oss.str());
                        }
                    } else {
                        // Normalized index is not an identifier (could be number or virtual index)
                        throw ParseError("Normalized index must be a free variable (lowercase identifier), not a number or virtual index");
                    }
                }
            }
        }

        // Constraint: at most one normalized index per equation
        if (normalizedCount > 1) {
            throw ParseError("Only one index can be normalized per equation");
        }
    }

    Statement parseStatement() {
        // Possible Datalog atom/rule/fact at statement start
        if (tok_.type == Token::Identifier && tok_.text != "file" && toks_.peek().type == Token::LParen && startsWithUpper(tok_.text)) {
            DatalogAtom head = parseAtom();
            // Datalog query: Atom? or Atom, body...? (conjunctive query with comparisons)
            if (accept(Token::Question)) {
                Query q; q.target = head; q.loc = head.loc;
                q.directive = parseQueryDirective();
                return q;
            }
            if (accept(Token::LArrow)) {
                // parse body: elements are either Atom, Negation, or Condition (comparison of tensor expressions)
                std::vector<std::variant<DatalogAtom, DatalogNegation, DatalogCondition>> body;
                body.push_back(parseRuleBodyElement());
                while (accept(Token::Comma)) body.push_back(parseRuleBodyElement());
                DatalogRule r; r.head = std::move(head); r.body = std::move(body); r.loc = r.head.loc;
                return r;
            }
            // Datalog conjunctive query: Atom, (Atom|Negation|Condition) ... ?
            if (tok_.type == Token::Comma) {
                std::vector<std::variant<DatalogAtom, DatalogNegation, DatalogCondition>> conj;
                // include the first atom
                conj.push_back(head);
                do {
                    advance(); // consume ','
                    conj.push_back(parseRuleBodyElement());
                } while (tok_.type == Token::Comma);
                expect(Token::Question, "'?' to end query");
                Query q; q.target = head; q.body = std::move(conj); q.loc = head.loc;
                q.directive = parseQueryDirective();
                return q;
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
        // Fallback: tensor equation or query
        TensorRef lhs = parseTensorRef();
        // Check if this is a query: tensor_ref?
        if (accept(Token::Question)) {
            Query q; q.target = lhs; q.loc = lhs.loc;
            q.directive = parseQueryDirective();
            return q;
        }
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
        validateNormalizedIndices(eq);
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

