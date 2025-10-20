#include "TL/Lexer.hpp"
#include <tao/pegtl.hpp>

namespace tl::lex {

    namespace pegtl = tao::pegtl;

    static SourceLocation locFrom(const pegtl::position& p) {
        SourceLocation l; l.line = p.line; l.column = p.column; return l;
    }

    // Whitespace and comments (skipped)
    struct sp : pegtl::sor< pegtl::one<' '>, pegtl::one<'\t'>, pegtl::one<'\r'> > {};
    struct line_comment : pegtl::seq< pegtl::two<'/'>, pegtl::until< pegtl::at< pegtl::eolf >, pegtl::any > > {};
    struct block_comment : pegtl::seq< pegtl::string<'/','*'>, pegtl::until< pegtl::string<'*','/'>, pegtl::any > > {};
    struct skipped : pegtl::sor< sp, line_comment, block_comment > {};

    // Newline token
    struct newline_tok : pegtl::one<'\n'> {};

    // Identifiers
    struct ident_start : pegtl::sor< pegtl::alpha, pegtl::one<'_'> > {};
    struct ident_rest  : pegtl::sor< pegtl::alnum, pegtl::one<'_'> > {};
    struct identifier  : pegtl::seq< ident_start, pegtl::star< ident_rest > > {};

    // Numbers (simplified: optional '-', digits, optional fraction or exponent)
    struct digits : pegtl::plus< pegtl::digit > {};
    struct opt_sign : pegtl::opt< pegtl::one<'-'> > {};
    struct frac : pegtl::seq< pegtl::one<'.'>, pegtl::star< pegtl::digit > > {};
    struct expn : pegtl::seq< pegtl::sor< pegtl::one<'e'>, pegtl::one<'E'> >, pegtl::opt< pegtl::one<'+','-'> >, digits > {};
    struct number : pegtl::seq< opt_sign, pegtl::sor< pegtl::seq< digits, pegtl::opt< frac > >, pegtl::seq< pegtl::one<'.'>, digits > >, pegtl::opt< expn > > {};

    // Strings: simple handling with escapes
    struct esc_seq : pegtl::seq< pegtl::one<'\\'>, pegtl::any > {};
    struct dquot_str_content : pegtl::until< pegtl::one<'"'>, pegtl::sor< esc_seq, pegtl::not_one<'"'> > > {};
    struct squot_str_content : pegtl::until< pegtl::one<'\''>, pegtl::sor< esc_seq, pegtl::not_one<'\''> > > {};
    struct dquoted_string : pegtl::seq< pegtl::one<'"'>, dquot_str_content > {};
    struct squoted_string : pegtl::seq< pegtl::one<'\''>, squot_str_content > {};
    struct string_lit : pegtl::sor< dquoted_string, squoted_string > {};

    // Single and multi-char tokens
    struct lbrack : pegtl::one<'['> {};
    struct rbrack : pegtl::one<']'> {};
    struct lparen : pegtl::one<'('> {};
    struct rparen : pegtl::one<')'> {};
    struct comma  : pegtl::one<','> {};
    struct equals : pegtl::one<'='> {};
    struct qmark  : pegtl::one<'?'> {};
    struct plus   : pegtl::one<'+'> {};
    struct minus  : pegtl::one<'-'> {};
    struct dot    : pegtl::one<'.'> {};
    struct slash  : pegtl::one<'/'> {};
    struct larrow : pegtl::string<'<','-'> {};

    // Comparison operators
    struct ge : pegtl::string<'>','='> {}; // >=
    struct le : pegtl::string<'<','='> {}; // <=
    struct eqeq : pegtl::string<'=','='> {}; // ==
    struct noteq : pegtl::string<'!','='> {}; // !=
    struct gt : pegtl::one<'>'> {};
    struct lt : pegtl::one<'<' > {};

    // Unknown single char fallback
    struct unknown_char : pegtl::not_one< '\0' > {};

    // Token union in priority order
    struct token_rule : pegtl::sor<
                newline_tok,
                skipped,
                string_lit,
                number,
                identifier,
                larrow, ge, le, eqeq, noteq, gt, lt,
                lbrack, rbrack, lparen, rparen, comma, equals, qmark, plus, minus, dot, slash,
                unknown_char
            > {};

    struct tokens_grammar : pegtl::must< pegtl::star< token_rule > > {};

    // Actions
    template< typename Rule > struct action : pegtl::nothing< Rule > {};

    struct TokenSink {
        std::vector<Token> out;
    };

    static std::string unescape(const std::string& s) {
        if (s.empty()) return {};
        char quote = s.front();
        size_t i = 1;
        std::string res;
        while (i < s.size()) {
            char c = s[i++];
            if (c == quote) break;
            if (c == '\\' && i < s.size()) {
                switch (const char e = s[i++]) {
                    case 'n': res.push_back('\n'); break;
                    case 't': res.push_back('\t'); break;
                    case 'r': res.push_back('\r'); break;
                    case '\\': res.push_back('\\'); break;
                    case '"': res.push_back('"'); break;
                    case '\'': res.push_back('\''); break;
                    case '0': res.push_back('\0'); break;
                    default: res.push_back(e); break;
                }
            } else {
                res.push_back(c);
            }
        }
        return res;
    }

    template<> struct action< newline_tok > {
        template< typename Input >
        static void apply(const Input& in, TokenSink& sink) {
            Token t; t.type = Token::Newline; t.text = "\n"; t.loc = locFrom(in.position());
            sink.out.push_back(std::move(t));
        }
    };

    // identifier
    template<> struct action< identifier > {
        template< typename Input >
        static void apply(const Input& in, TokenSink& sink) {
            Token t; t.type = Token::Identifier; t.text = in.string(); t.loc = locFrom(in.position());
            sink.out.push_back(std::move(t));
        }
    };

    // number
    template<> struct action< number > {
        template< typename Input >
        static void apply(const Input& in, TokenSink& sink) {
            Token t; t.text = in.string(); t.loc = locFrom(in.position());
            if (t.text.find_first_of(".eE") != std::string::npos) t.type = Token::Float; else t.type = Token::Integer;
            sink.out.push_back(std::move(t));
        }
    };

    // string
    template<> struct action< string_lit > {
        template< typename Input >
        static void apply(const Input& in, TokenSink& sink) {
            Token t; t.type = Token::String; t.text = unescape(in.string()); t.loc = locFrom(in.position());
            sink.out.push_back(std::move(t));
        }
    };

#define DEFINE_CHAR_TOKEN(rule_name, token_type, ch) \
    template<> struct action< rule_name > { \
        template< typename Input > \
        static void apply(const Input& in, TokenSink& sink) { \
            Token t; t.type = token_type; t.text = std::string(1, ch); t.loc = locFrom(in.position()); \
            sink.out.push_back(std::move(t)); \
        } \
    };

    DEFINE_CHAR_TOKEN(lbrack, Token::LBracket, '[')
    DEFINE_CHAR_TOKEN(rbrack, Token::RBracket, ']')
    DEFINE_CHAR_TOKEN(lparen, Token::LParen, '(')
    DEFINE_CHAR_TOKEN(rparen, Token::RParen, ')')
    DEFINE_CHAR_TOKEN(comma,  Token::Comma,  ',')
    DEFINE_CHAR_TOKEN(equals, Token::Equals, '=')
    DEFINE_CHAR_TOKEN(qmark,  Token::Question, '?')
    DEFINE_CHAR_TOKEN(plus,   Token::Plus,   '+')
    DEFINE_CHAR_TOKEN(minus,  Token::Minus,  '-')
    DEFINE_CHAR_TOKEN(dot,    Token::Dot,    '.')
    DEFINE_CHAR_TOKEN(slash,  Token::Slash,  '/')

    // multi-char '<-'
    template<> struct action< larrow > {
        template< typename Input >
        static void apply(const Input& in, TokenSink& sink) {
            Token t; t.type = Token::LArrow; t.text = in.string(); t.loc = locFrom(in.position());
            sink.out.push_back(std::move(t));
        }
    };

    // comparison actions
#define DEFINE_TOKEN_ACTION(rule, tokentype) \
    template<> struct action< rule > { \
        template< typename Input > \
        static void apply(const Input& in, TokenSink& sink) { \
            Token t; t.type = tokentype; t.text = in.string(); t.loc = locFrom(in.position()); \
            sink.out.push_back(std::move(t)); \
        } \
    };

    DEFINE_TOKEN_ACTION(ge, Token::Ge)
    DEFINE_TOKEN_ACTION(le, Token::Le)
    DEFINE_TOKEN_ACTION(eqeq, Token::EqEq)
    DEFINE_TOKEN_ACTION(noteq, Token::NotEq)
    DEFINE_TOKEN_ACTION(gt, Token::Greater)
    DEFINE_TOKEN_ACTION(lt, Token::Less)

    // Unknown
    template<> struct action< unknown_char > {
        template< typename Input >
        static void apply(const Input& in, TokenSink& sink) {
            Token t; t.type = Token::Unknown; t.text = in.string(); t.loc = locFrom(in.position());
            sink.out.push_back(std::move(t));
        }
    };

    TokenStream::TokenStream(std::string_view src) {
        pegtl::memory_input in(src, "<input>");
        TokenSink sink;
        pegtl::parse< tokens_grammar, action >(in, sink);
        Token end; end.type = Token::End; end.text = ""; end.loc = {0,0};
        sink.out.push_back(std::move(end));
        tokens_ = std::move(sink.out);
    }

    const Token& TokenStream::peek() const { return tokens_[idx_]; }
    const Token& TokenStream::lookahead(size_t n) const { return tokens_[idx_ + n]; }
    Token TokenStream::consume() { return tokens_[idx_++]; }

}
