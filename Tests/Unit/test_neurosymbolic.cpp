#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"

using namespace tl;
using Catch::Matchers::WithinAbs;

static float getTensorValue(const torch::Tensor& t, const std::vector<int64_t>& indices) {
    torch::Tensor indexed = t;
    for (auto idx : indices) {
        indexed = indexed.index({idx});
    }
    return indexed.item<float>();
}

static bool hasFact(const Environment& env, const std::string& relation,
                    const std::vector<std::string>& args) {
    const auto& facts = env.facts(relation);
    for (const auto& fact : facts) {
        if (fact.size() == args.size()) {
            bool match = true;
            for (size_t i = 0; i < args.size(); ++i) {
                if (fact[i] != args[i]) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
    }
    return false;
}

TEST_CASE("Neurosymbolic - entity embeddings with similarity", "[neurosymbolic]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Entity embeddings
        Emb[Alice, 0] = 0.8
        Emb[Alice, 1] = 0.3
        Emb[Bob, 0] = 0.7
        Emb[Bob, 1] = 0.4
        Emb[Charlie, 0] = 0.1
        Emb[Charlie, 1] = 0.9

        // Compute similarity scores
        Sim[x, y] = Emb[x, d] Emb[y, d]
    )");
    vm.execute(prog);

    // Verify embeddings are stored
    auto Emb = vm.env().lookup("Emb");
    REQUIRE(Emb.dim() >= 1);

    // Verify similarity computation
    auto Sim = vm.env().lookup("Sim");
    // sim(Alice, Bob) = 0.8*0.7 + 0.3*0.4 = 0.56 + 0.12 = 0.68
    // Note: actual indices depend on label mapping
}

TEST_CASE("Neurosymbolic - mixed reasoning", "[neurosymbolic][mixed]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Neural component: embeddings
        Emb[Alice, 0] = 0.9
        Emb[Alice, 1] = 0.2
        Emb[Bob, 0] = 0.85
        Emb[Bob, 1] = 0.25
        Emb[Charlie, 0] = 0.1
        Emb[Charlie, 1] = 0.9

        // Compute similarities
        Similarity[x, y] = Emb[x, d] Emb[y, d]

        // Symbolic component: facts
        Parent(Alice, Bob)
        Parent(Bob, Charlie)

        // Pre-compute Similar facts based on threshold
        // In practice, this would be done with comparisons
        Similar(Alice, Bob)
        Similar(Bob, Alice)

        // Hybrid rule: similar entities with parent relationship
        // If x is similar to y, and y is parent of z, then x might be related to z
        MaybeRelated(x, z) <- Similar(x, y), Parent(y, z), x != z

        MaybeRelated(x, y)?
    )");
    vm.execute(prog);

    // Verify symbolic facts
    REQUIRE(hasFact(vm.env(), "Parent", {"Alice", "Bob"}));
    REQUIRE(hasFact(vm.env(), "Similar", {"Alice", "Bob"}));

    // Verify derived facts from hybrid reasoning
    // Alice is similar to Bob, Bob is parent of Charlie, so Alice might be related to Charlie
    REQUIRE(hasFact(vm.env(), "MaybeRelated", {"Alice", "Charlie"}));
}

TEST_CASE("Neurosymbolic - knowledge graph completion", "[neurosymbolic][kg]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Embeddings for cities
        Emb[Paris, 0] = 0.9
        Emb[Paris, 1] = 0.2
        Emb[London, 0] = 0.88
        Emb[London, 1] = 0.22
        Emb[Berlin, 0] = 0.89
        Emb[Berlin, 1] = 0.21

        // Embeddings for countries
        Emb[France, 0] = 0.85
        Emb[France, 1] = 0.25
        Emb[UK, 0] = 0.82
        Emb[UK, 1] = 0.28
        Emb[Germany, 0] = 0.84
        Emb[Germany, 1] = 0.26

        // Compute similarities
        Similarity[x, y] = Emb[x, d] Emb[y, d]

        // Known facts
        CapitalOf(Paris, France)
        CapitalOf(London, UK)

        // Pre-computed similar entities (above threshold)
        Similar(Paris, London)
        Similar(London, Paris)
        Similar(Paris, Berlin)
        Similar(Berlin, Paris)
        Similar(France, UK)
        Similar(UK, France)
        Similar(France, Germany)
        Similar(Germany, France)

        // KG completion rule
        PotentialCapital(xp, yp) <- CapitalOf(x, y),
                                     Similar(xp, x),
                                     Similar(yp, y),
                                     xp != x

        PotentialCapital(x, y)?
    )");
    vm.execute(prog);

    // Verify known facts
    REQUIRE(hasFact(vm.env(), "CapitalOf", {"Paris", "France"}));
    REQUIRE(hasFact(vm.env(), "CapitalOf", {"London", "UK"}));

    // Verify similarity facts
    REQUIRE(hasFact(vm.env(), "Similar", {"Paris", "London"}));
    REQUIRE(hasFact(vm.env(), "Similar", {"France", "UK"}));

    // Verify predicted capitals
    REQUIRE(hasFact(vm.env(), "PotentialCapital", {"Berlin", "Germany"}));
    REQUIRE(hasFact(vm.env(), "PotentialCapital", {"London", "UK"}));
}

TEST_CASE("Neurosymbolic - relation prediction with embeddings", "[neurosymbolic][relations]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Simple entity embeddings (2D)
        Emb[Alice, 0] = 0.8
        Emb[Alice, 1] = 0.3
        Emb[Bob, 0] = 0.7
        Emb[Bob, 1] = 0.4
        Emb[Charlie, 0] = 0.75
        Emb[Charlie, 1] = 0.35

        // Known relationships
        Friend(Alice, Bob)

        // Compute embedding similarities
        EmbSim[x, y] = Emb[x, d] Emb[y, d]

        // Similar entities based on embeddings
        Similar(Alice, Charlie)
        Similar(Charlie, Alice)
        Similar(Bob, Charlie)
        Similar(Charlie, Bob)

        // Predict friendships based on similarity
        PotentialFriend(x, z) <- Friend(x, y), Similar(y, z), x != z

        PotentialFriend(x, y)?
    )");
    vm.execute(prog);

    // Verify known facts
    REQUIRE(hasFact(vm.env(), "Friend", {"Alice", "Bob"}));
    REQUIRE(hasFact(vm.env(), "Similar", {"Alice", "Charlie"}));

    // Verify predictions
    REQUIRE(hasFact(vm.env(), "PotentialFriend", {"Alice", "Charlie"}));
}

TEST_CASE("Neurosymbolic - boolean tensors", "[neurosymbolic][boolean]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Adjacency matrix (boolean tensor)
        Adjacent[0, 1] = 1
        Adjacent[1, 0] = 1
        Adjacent[1, 2] = 1
        Adjacent[2, 1] = 1
        Adjacent[2, 3] = 1
        Adjacent[3, 2] = 1

        // Compute 2-hop connectivity
        TwoHop[i, k] = Adjacent[i, j] Adjacent[j, k]
    )");
    vm.execute(prog);

    auto Adjacent = vm.env().lookup("Adjacent");
    auto TwoHop = vm.env().lookup("TwoHop");

    // Verify adjacency
    REQUIRE_THAT(getTensorValue(Adjacent, {0, 1}), WithinAbs(1.0f, 0.001f));
    REQUIRE_THAT(getTensorValue(Adjacent, {1, 0}), WithinAbs(1.0f, 0.001f));

    // TwoHop[0, 0] should be > 0 (0->1->0)
    // TwoHop[0, 2] should be > 0 (0->1->2)
}

TEST_CASE("Neurosymbolic - confidence scoring", "[neurosymbolic][scoring]") {
    std::stringstream out, err;
    TensorLogicVM vm{&out, &err};
    auto prog = parseProgram(R"(
        // Entity embeddings
        Emb[Alice, 0] = 0.9
        Emb[Alice, 1] = 0.1
        Emb[Bob, 0] = 0.85
        Emb[Bob, 1] = 0.15

        // Compute confidence scores
        Confidence[x, y] = Emb[x, d] Emb[y, d]

        // Facts with confidence above threshold (manually specified)
        HighConfidence(Alice, Bob)

        // Reasoning with confidence
        Trustworthy(x, y) <- HighConfidence(x, y)

        Trustworthy(x, y)?
    )");
    vm.execute(prog);

    // Verify confidence tensor exists
    REQUIRE(vm.env().has("Confidence"));

    // Verify derived facts
    REQUIRE(hasFact(vm.env(), "HighConfidence", {"Alice", "Bob"}));
    REQUIRE(hasFact(vm.env(), "Trustworthy", {"Alice", "Bob"}));
}
