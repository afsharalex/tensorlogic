#include <catch2/catch_test_macros.hpp>
#include "TL/Parser.hpp"
#include "TL/AST.hpp"
#include <string>

using namespace tl;

static std::string stmtStr(const Program& p, size_t i) {
    REQUIRE(i < p.statements.size());
    return toString(p.statements[i]);
}

TEST_CASE("Parse Examples/ParseTests/00_min.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/00_min.tl");
    REQUIRE(p.statements.size() >= 1);
}

TEST_CASE("Parse Examples/ParseTests/01_simple_equation.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/01_simple_equation.tl");
    REQUIRE(p.statements.size() == 1);
}

TEST_CASE("Parse Examples/ParseTests/02_matrix_multiply.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/02_matrix_multiply.tl");
    REQUIRE(p.statements.size() == 1);
    CHECK(stmtStr(p,0) == "Y[i,k] = W[i,j]X[j,k]");
}

TEST_CASE("Parse Examples/ParseTests/03_vector_product.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/03_vector_product.tl");
    REQUIRE(p.statements.size() >= 1);
}

TEST_CASE("Parse Examples/ParseTests/04_function_calls.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/04_function_calls.tl");
    REQUIRE(p.statements.size() == 5);
    CHECK(stmtStr(p,0) == "Y1[i] = sigmoid(X[i])");
}

TEST_CASE("Parse Examples/ParseTests/07_arithmetic.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/07_arithmetic.tl");
    REQUIRE(p.statements.size() == 3);
    CHECK(stmtStr(p,0) == "Y1[i] = X[i]+b[i]");
    CHECK(stmtStr(p,1) == "Y2[i] = X[i]-mean");
    CHECK(stmtStr(p,2) == "Y3[i] = X[i]/Z");
}

TEST_CASE("Parse Examples/ParseTests/08_datalog_facts.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/08_datalog_facts.tl");
    REQUIRE(p.statements.size() == 4);
    CHECK(stmtStr(p,0) == "Parent(Alice,Bob)");
}

TEST_CASE("Parse Examples/ParseTests/09_datalog_rules.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/09_datalog_rules.tl");
    REQUIRE(p.statements.size() == 2);
    CHECK(stmtStr(p,0) == "Ancestor(x,y) <- Parent(x,y)");
}

TEST_CASE("Parse Examples/ParseTests/10_datalog_query.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/10_datalog_query.tl");
    REQUIRE(p.statements.size() == 5);
    CHECK(stmtStr(p,4) == "Ancestor(x,Charlie)?");
}

TEST_CASE("Parse Examples/ParseTests/14_mixed_neurosymbolic.tl") {
    auto p = parseFile(std::string(TL_SOURCE_DIR) + "/Examples/ParseTests/14_mixed_neurosymbolic.tl");
    REQUIRE(p.statements.size() == 5);
    CHECK(stmtStr(p,3) == "Similar(x,y) <- Emb[x,d]Emb[y,d] > threshold");
    CHECK(stmtStr(p,4) == "MaybeRelated(x,z) <- Similar(x,y),Parent(y,z)");
}
