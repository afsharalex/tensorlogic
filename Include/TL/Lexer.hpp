#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "TL/AST.hpp"

namespace tl::lex {

    struct Token {
        enum Type {
            Identifier, Integer, Float, String,
            LBracket, RBracket, LParen, RParen,
            Comma, Equals, Question,
            Plus, Minus, Dot, Slash,
            Greater, Less, Ge, Le, EqEq, NotEq,
            LArrow, // '<-' for Datalog rules
            End, Newline, Unknown
        } type{End};
        std::string text;
        SourceLocation loc{};
    };

    class TokenStream {
    public:
        explicit TokenStream(std::string_view src);
        [[nodiscard]] const Token& peek() const;
        [[nodiscard]] const Token& lookahead(size_t n) const; // look ahead without consuming
        Token consume();
    private:
        std::vector<Token> tokens_;
        size_t idx_{0};
    };

}
