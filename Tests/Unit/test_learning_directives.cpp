#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "TL/Parser.hpp"
#include "TL/vm.hpp"
#include <sstream>

using namespace tl;

TEST_CASE("Parse learning directive - minimize", "[learning][parse]") {
    std::string code = R"(
        Loss = [1.0]
        Loss? @minimize(lr=0.01, epochs=100)
    )";

    Program prog = parseProgram(code);
    REQUIRE(prog.statements.size() == 2);

    auto* query = std::get_if<Query>(&prog.statements[1]);
    REQUIRE(query != nullptr);
    REQUIRE(query->directive.has_value());

    const auto& dir = query->directive.value();
    REQUIRE(dir.name.name == "minimize");
    REQUIRE(dir.args.size() == 2);

    // Check lr argument
    REQUIRE(dir.args[0].name.name == "lr");
    auto* lr_num = std::get_if<NumberLiteral>(&dir.args[0].value);
    REQUIRE(lr_num != nullptr);
    REQUIRE(lr_num->text == "0.01");

    // Check epochs argument
    REQUIRE(dir.args[1].name.name == "epochs");
    auto* epochs_num = std::get_if<NumberLiteral>(&dir.args[1].value);
    REQUIRE(epochs_num != nullptr);
    REQUIRE(epochs_num->text == "100");
}

TEST_CASE("Parse learning directive - maximize", "[learning][parse]") {
    std::string code = R"(
        Reward = [1.0]
        Reward? @maximize(lr=0.05, epochs=50, verbose=true)
    )";

    Program prog = parseProgram(code);
    REQUIRE(prog.statements.size() == 2);

    auto* query = std::get_if<Query>(&prog.statements[1]);
    REQUIRE(query != nullptr);
    REQUIRE(query->directive.has_value());

    const auto& dir = query->directive.value();
    REQUIRE(dir.name.name == "maximize");
    REQUIRE(dir.args.size() == 3);

    // Check verbose argument (boolean)
    REQUIRE(dir.args[2].name.name == "verbose");
    auto* verbose_bool = std::get_if<bool>(&dir.args[2].value);
    REQUIRE(verbose_bool != nullptr);
    REQUIRE(*verbose_bool == true);
}

TEST_CASE("Parse learning directive - sample", "[learning][parse]") {
    std::string code = R"(
        Probs = [0.5, 1.0, 2.0]
        Probs? @sample(n=1000)
    )";

    Program prog = parseProgram(code);
    REQUIRE(prog.statements.size() == 2);

    auto* query = std::get_if<Query>(&prog.statements[1]);
    REQUIRE(query != nullptr);
    REQUIRE(query->directive.has_value());

    const auto& dir = query->directive.value();
    REQUIRE(dir.name.name == "sample");
    REQUIRE(dir.args.size() == 1);
    REQUIRE(dir.args[0].name.name == "n");
}

TEST_CASE("LearningConfig from directive", "[learning]") {
    std::string code = R"(
        Loss = [1.0]
        Loss? @minimize(lr=0.02, epochs=200, verbose=true)
    )";

    Program prog = parseProgram(code);
    auto* query = std::get_if<Query>(&prog.statements[1]);
    REQUIRE(query != nullptr);
    REQUIRE(query->directive.has_value());

    LearningConfig config = LearningConfig::fromDirective(query->directive.value());
    REQUIRE(config.directive == "minimize");
    REQUIRE(config.learningRate == 0.02);
    REQUIRE(config.epochs == 200);
    REQUIRE(config.verbose == true);
}

TEST_CASE("Identify learnable parameters - simple", "[learning]") {
    std::string code = R"(
        W = [[0.5, 0.3], [0.2, 0.8]]
        X = [1.0, 0.5]
        Y[i] = W[i, j] X[j]
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    // Execute to populate environment
    vm.execute(prog);

    // Check that W and X are in environment
    REQUIRE(vm.env().has("W"));
    REQUIRE(vm.env().has("X"));
    REQUIRE(vm.env().has("Y"));
}

TEST_CASE("Query without directive works", "[learning][query]") {
    std::string code = R"(
        X = [1.0, 2.0, 3.0]
        X?
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);
    vm.execute(prog);

    std::string output = out.str();
    REQUIRE(output.find("1") != std::string::npos);
    REQUIRE(output.find("2") != std::string::npos);
    REQUIRE(output.find("3") != std::string::npos);
}

// Gradient tracking tests - these now work correctly!

TEST_CASE("Simple quadratic optimization", "[learning][minimize]") {
    // Test case: minimize (x - 2)^2
    // Optimal solution: x = 2
    std::string code = R"(
        x = [0.0]
        Target = [2.0]
        diff = x[0] - Target[0]
        loss = diff^2
        loss? @minimize(lr=0.1, epochs=100)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    REQUIRE_NOTHROW(vm.execute(prog));

    // After optimization, x should be close to 2.0
    const auto& x = vm.env().lookup("x");
    double x_val = x[0].item<double>();
    REQUIRE_THAT(x_val, Catch::Matchers::WithinAbs(2.0, 0.1));
}

TEST_CASE("Linear regression convergence", "[learning][minimize]") {
    // y = 2x + 1, learn m and b
    std::string code = R"(
        X = [1.0, 2.0, 3.0, 4.0, 5.0]
        Y = [3.0, 5.0, 7.0, 9.0, 11.0]

        m = [0.5]
        b = [0.0]

        Pred[i] = m[0] X[i] + b[0]
        Err[i] = (Pred[i] - Y[i])^2
        Loss = Err[i]

        Loss? @minimize(lr=0.01, epochs=200)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    REQUIRE_NOTHROW(vm.execute(prog));

    // After optimization, m should be close to 2.0, b close to 1.0
    const auto& m = vm.env().lookup("m");
    const auto& b = vm.env().lookup("b");

    double m_val = m[0].item<double>();
    double b_val = b[0].item<double>();

    REQUIRE_THAT(m_val, Catch::Matchers::WithinAbs(2.0, 0.2));
    REQUIRE_THAT(b_val, Catch::Matchers::WithinAbs(1.0, 0.2));
}

TEST_CASE("Maximize simple reward", "[learning][maximize]") {
    // Maximize -(x - 3)^2, optimal at x = 3
    std::string code = R"(
        x = [0.0]
        Target = [3.0]
        diff = x[0] - Target[0]
        neg_loss = -(diff^2)
        neg_loss? @maximize(lr=0.1, epochs=100)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    REQUIRE_NOTHROW(vm.execute(prog));

    const auto& x = vm.env().lookup("x");
    double x_val = x[0].item<double>();
    REQUIRE_THAT(x_val, Catch::Matchers::WithinAbs(3.0, 0.1));
}

TEST_CASE("Sample from distribution", "[learning][sample]") {
    std::string code = R"(
        Probs = [1.0, 2.0, 3.0, 4.0]
        Probs? @sample(n=100)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    REQUIRE_NOTHROW(vm.execute(prog));

    // Output should contain sample results
    std::string output = out.str();
    REQUIRE(!output.empty());
}

TEST_CASE("Verbose mode outputs progress", "[learning][verbose]") {
    std::string code = R"(
        x = [0.0]
        loss = x[0]^2
        loss? @minimize(lr=0.1, epochs=10, verbose=true)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    REQUIRE_NOTHROW(vm.execute(prog));

    std::string output = out.str();
    // Should contain "Epoch" in verbose output
    REQUIRE(output.find("Epoch") != std::string::npos);
}

TEST_CASE("Multi-parameter optimization", "[learning][minimize]") {
    // Minimize (x - 1)^2 + (y - 2)^2
    // Optimal: x = 1, y = 2
    std::string code = R"(
        x = [0.0]
        y = [0.0]

        TX = [1.0]
        TY = [2.0]

        dx = x[0] - TX[0]
        dy = y[0] - TY[0]

        loss = dx^2 + dy^2
        loss? @minimize(lr=0.1, epochs=150)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    REQUIRE_NOTHROW(vm.execute(prog));

    const auto& x = vm.env().lookup("x");
    const auto& y = vm.env().lookup("y");

    double x_val = x[0].item<double>();
    double y_val = y[0].item<double>();

    REQUIRE_THAT(x_val, Catch::Matchers::WithinAbs(1.0, 0.1));
    REQUIRE_THAT(y_val, Catch::Matchers::WithinAbs(2.0, 0.1));
}

TEST_CASE("Error: No learnable parameters", "[learning][error]") {
    // All tensors are uppercase (data/constants), none are learnable by naming convention
    // Per the heuristic: lowercase = parameter, UPPERCASE = data (except W*)
    std::string code = R"(
        X = [1.0]
        Y = X[0] + 1.0
        Y? @minimize(lr=0.1, epochs=10)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    // Should throw error about no learnable parameters
    // X and Y are uppercase (data), so neither should be learnable
    REQUIRE_THROWS_WITH(vm.execute(prog),
        Catch::Matchers::ContainsSubstring("No learnable parameters"));
}

TEST_CASE("Error: Missing target tensor", "[learning][error]") {
    std::string code = R"(
        x = [1.0]
        NonExistent? @minimize(lr=0.1, epochs=10)
    )";

    Program prog = parseProgram(code);

    std::ostringstream out, err;
    TensorLogicVM vm(&out, &err);

    // Should throw error about missing tensor
    REQUIRE_THROWS_WITH(vm.execute(prog),
        Catch::Matchers::ContainsSubstring("not found"));
}

TEST_CASE("Default parameter values", "[learning]") {
    std::string code = R"(
        x = [1.0]
        loss = x[0]^2
        loss? @minimize()
    )";

    Program prog = parseProgram(code);
    auto* query = std::get_if<Query>(&prog.statements[2]);
    REQUIRE(query != nullptr);
    REQUIRE(query->directive.has_value());

    LearningConfig config = LearningConfig::fromDirective(query->directive.value());

    // Check default values
    REQUIRE(config.learningRate == 0.01);
    REQUIRE(config.epochs == 100);
    REQUIRE(config.sampleCount == 1000);
    REQUIRE(config.verbose == false);
}

TEST_CASE("Alternative argument names", "[learning][parse]") {
    // Test lr vs learning_rate, n vs samples
    std::string code1 = R"(
        x = [1.0]
        x? @minimize(learning_rate=0.05)
    )";

    Program prog1 = parseProgram(code1);
    auto* query1 = std::get_if<Query>(&prog1.statements[1]);
    REQUIRE(query1 != nullptr);

    LearningConfig config1 = LearningConfig::fromDirective(query1->directive.value());
    REQUIRE(config1.learningRate == 0.05);

    std::string code2 = R"(
        p = [1.0, 2.0]
        p? @sample(samples=500)
    )";

    Program prog2 = parseProgram(code2);
    auto* query2 = std::get_if<Query>(&prog2.statements[1]);
    REQUIRE(query2 != nullptr);

    LearningConfig config2 = LearningConfig::fromDirective(query2->directive.value());
    REQUIRE(config2.sampleCount == 500);
}
