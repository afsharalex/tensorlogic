#include <catch2/catch_test_macros.hpp>
#include "TL/Parser.hpp"
#include "TL/AST.hpp"
#include <string>

using namespace tl;

static std::string first(const Program& p) {
    REQUIRE_FALSE(p.statements.empty());
    return toString(p.statements.front());
}

TEST_CASE("Tensor equation: simple add/sub/mul/div") {
    {
        auto p = parseProgram("Y[i] = X[i] + b[i]\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Y[i] = X[i]+b[i]");
    }
    {
        auto p = parseProgram("Y[i] = X[i] - mean\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Y[i] = X[i]-mean");
    }
    {
        auto p = parseProgram("Y[i,k] = W[i,j] X[j,k]\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Y[i,k] = W[i,j]X[j,k]");
    }
    {
        auto p = parseProgram("Y[i] = X[i] / Z\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Y[i] = X[i]/Z");
    }
}

TEST_CASE("Function calls and normalized indices") {
    {
        auto p = parseProgram("Y[i] = sigmoid(X[i])\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Y[i] = sigmoid(X[i])");
    }
    {
        auto p = parseProgram("Y[i.] = softmax(X[i])\n");
        REQUIRE(p.statements.size() == 1);
        // Normalized index is syntactic only currently; prints without the dot in indices
        CHECK(first(p) == "Y[i] = softmax(X[i])");
    }
}

TEST_CASE("File operations: file(\"...\") = T and \"...\" = T") {
    {
        auto p = parseProgram("file(\"/tmp/out.txt\") = A[i]\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "\"/tmp/out.txt\" = A[i]");
    }
    {
        auto p = parseProgram("\"/tmp/out2.txt\" = B[j]\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "\"/tmp/out2.txt\" = B[j]");
    }
}

TEST_CASE("Queries: tensor ref and Datalog atom") {
    {
        auto p = parseProgram("A[i]?\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "A[i]?");
    }
    {
        auto p = parseProgram("Ancestor(x,Charlie)?\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Ancestor(x,Charlie)?");
    }
}

TEST_CASE("Datalog facts") {
    auto p = parseProgram("Parent(Alice,Bob)\n");
    REQUIRE(p.statements.size() == 1);
    CHECK(first(p) == "Parent(Alice,Bob)");
}

TEST_CASE("Datalog rules: simple and mixed neurosymbolic condition") {
    {
        auto p = parseProgram("Ancestor(x,y) <- Parent(x,y)\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Ancestor(x,y) <- Parent(x,y)");
    }
    {
        auto p = parseProgram("Similar(x,y) <- Emb[x,d] Emb[y,d] > threshold\n");
        REQUIRE(p.statements.size() == 1);
        CHECK(first(p) == "Similar(x,y) <- Emb[x,d]Emb[y,d] > threshold");
    }
}
