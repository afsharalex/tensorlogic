#include "TL/Runtime/LearningEngine.hpp"
#include "TL/vm.hpp"
#include "TL/backend.hpp"
#include "TL/Runtime/ExecutorRegistry.hpp"
#include "TL/Runtime/Executors/ScalarAssignExecutor.hpp"
#include "TL/Runtime/Executors/ListLiteralExecutor.hpp"
#include "TL/Runtime/Executors/EinsumExecutor.hpp"
#include "TL/Runtime/Executors/IndexedProductExecutor.hpp"
#include "TL/Runtime/Executors/ReductionExecutor.hpp"
#include "TL/Runtime/Executors/NormalizationExecutor.hpp"
#include "TL/Runtime/Executors/GuardedClauseExecutor.hpp"
#include "TL/Runtime/Executors/PoolingExecutor.hpp"
#include "TL/Runtime/Executors/IdentityExecutor.hpp"
#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace tl {

LearningConfig LearningConfig::fromDirective(const QueryDirective& dir) {
    LearningConfig config;
    config.directive = dir.name.name;

    for (const auto& arg : dir.args) {
        const std::string& name = arg.name.name;

        if (name == "lr" || name == "learning_rate") {
            if (auto* num = std::get_if<NumberLiteral>(&arg.value)) {
                config.learningRate = std::stod(num->text);
            }
        } else if (name == "epochs") {
            if (auto* num = std::get_if<NumberLiteral>(&arg.value)) {
                config.epochs = std::stoi(num->text);
            }
        } else if (name == "n" || name == "samples") {
            if (auto* num = std::get_if<NumberLiteral>(&arg.value)) {
                config.sampleCount = std::stoi(num->text);
            }
        } else if (name == "verbose") {
            if (auto* b = std::get_if<bool>(&arg.value)) {
                config.verbose = *b;
            }
        }
    }

    return config;
}

LearningEngine::LearningEngine(Environment& env, TensorBackend& backend, ExecutorRegistry& registry)
    : env_(env), backend_(backend), registry_(registry) {}

torch::Tensor LearningEngine::executeDirective(
    const std::string& targetName,
    const QueryDirective& directive,
    const Program& program
) {
    LearningConfig config = LearningConfig::fromDirective(directive);

    if (config.directive == "minimize") {
        return minimize(targetName, config, program);
    } else if (config.directive == "maximize") {
        return maximize(targetName, config, program);
    } else if (config.directive == "sample") {
        return sample(targetName, config);
    } else {
        throw std::runtime_error("Unknown learning directive: " + config.directive);
    }
}

std::unordered_set<std::string> LearningEngine::identifyLearnableParameters(const Program& program) {
    std::unordered_set<std::string> learnable;
    std::unordered_set<std::string> computed;

    // Identify which tensors are computed from others (have dependencies)
    // A tensor is learnable if it's initialized with a literal list
    for (const auto& stmt : program.statements) {
        if (auto* eq = std::get_if<TensorEquation>(&stmt)) {
            const std::string& lhsName = eq->lhs.name.name;

            // Check if RHS is a simple list literal assignment
            bool isListLiteral = false;
            if (eq->clauses.size() == 1) {
                const auto& clause = eq->clauses[0];
                if (auto* exprList = std::get_if<ExprList>(&clause.expr->node)) {
                    isListLiteral = true;
                }
            }

            if (isListLiteral) {
                // This is a learnable parameter initialized with a list
                learnable.insert(lhsName);
            } else {
                // This is a computed tensor
                computed.insert(lhsName);
            }
        }
    }

    return learnable;
}

void LearningEngine::enableGradients(const std::unordered_set<std::string>& params) {
    // This is now handled in minimize/maximize directly
    // Keeping this method for potential future use
    (void)params;  // Suppress unused parameter warning
}

void LearningEngine::forwardPass(const Program& program) {
    // Execute all tensor equations in order
    for (const auto& stmt : program.statements) {
        if (auto* eq = std::get_if<TensorEquation>(&stmt)) {
            // For learnable parameters (list literals), skip re-execution
            // They should retain their current values and gradients
            bool isListLiteral = false;
            if (eq->clauses.size() == 1) {
                if (auto* exprList = std::get_if<ExprList>(&eq->clauses[0].expr->node)) {
                    isListLiteral = true;
                }
            }

            if (isListLiteral) {
                // Skip - this is a learnable parameter, don't re-initialize
                continue;
            }

            // Execute the equation - this will compute fresh values with gradients
            registry_.execute(*eq, env_, backend_);
        }
    }
}

void LearningEngine::backwardPass(
    const std::string& targetName,
    const std::unordered_set<std::string>& params,
    torch::optim::Optimizer& optimizer
) {
    if (!env_.has(targetName)) {
        throw std::runtime_error("Target tensor not found: " + targetName);
    }

    torch::Tensor target = env_.lookup(targetName);

    // Ensure the target is a scalar or reduce to scalar
    torch::Tensor loss = target;
    if (loss.numel() > 1) {
        loss = loss.sum();  // Reduce to scalar if needed
    }

    // Compute gradients
    loss.backward();

    // Update parameters using optimizer
    optimizer.step();
}

void LearningEngine::zeroGradients(const std::unordered_set<std::string>& params) {
    for (const auto& name : params) {
        if (env_.has(name)) {
            Tensor t = env_.lookup(name);
            if (t.grad().defined()) {
                t.grad().zero_();
            }
        }
    }
}

torch::Tensor LearningEngine::minimize(
    const std::string& lossName,
    const LearningConfig& config,
    const Program& program
) {
    // Identify learnable parameters
    auto params = identifyLearnableParameters(program);

    if (params.empty()) {
        throw std::runtime_error("No learnable parameters found for minimization");
    }

    // Enable gradients for learnable parameters
    enableGradients(params);

    // Create parameter vector for optimizer
    // IMPORTANT: We need to pass the actual tensors that will be updated,
    // not copies. We'll get them from env after each forward pass.
    std::vector<torch::Tensor> param_tensors;
    for (const auto& name : params) {
        param_tensors.push_back(env_.lookup(name).clone().detach().requires_grad_(true));
    }

    // Rebind the parameters with gradients enabled
    size_t idx = 0;
    for (const auto& name : params) {
        env_.bind(name, param_tensors[idx]);
        if (config.verbose) {
            std::cout << "Parameter " << name << " requires_grad="
                      << param_tensors[idx].requires_grad() << std::endl;
        }
        idx++;
    }

    // Create SGD optimizer
    torch::optim::SGD optimizer(param_tensors, torch::optim::SGDOptions(config.learningRate));

    // Training loop
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        forwardPass(program);

        // Get current loss
        if (!env_.has(lossName)) {
            throw std::runtime_error("Loss tensor not found: " + lossName);
        }

        torch::Tensor loss = env_.lookup(lossName);
        if (config.verbose && epoch == 0) {
            std::cout << "Loss tensor requires_grad=" << loss.requires_grad()
                      << ", has grad_fn=" << (loss.grad_fn() != nullptr) << std::endl;
        }

        if (loss.numel() > 1) {
            loss = loss.sum();
        }

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step();

        // Print progress
        if (config.verbose && (epoch % (config.epochs / 10) == 0 || epoch == config.epochs - 1)) {
            std::cout << "Epoch " << epoch << "/" << config.epochs
                      << " - Loss: " << loss.item<double>() << std::endl;
        }
    }

    // Return final loss value
    return env_.lookup(lossName).detach();
}

torch::Tensor LearningEngine::maximize(
    const std::string& rewardName,
    const LearningConfig& config,
    const Program& program
) {
    // Maximizing reward is equivalent to minimizing negative reward
    // We'll temporarily negate the reward tensor

    auto params = identifyLearnableParameters(program);
    if (params.empty()) {
        throw std::runtime_error("No learnable parameters found for maximization");
    }

    // Create parameter vector for optimizer with gradients enabled
    std::vector<torch::Tensor> param_tensors;
    for (const auto& name : params) {
        param_tensors.push_back(env_.lookup(name).clone().detach().requires_grad_(true));
    }

    // Rebind the parameters with gradients enabled
    size_t idx = 0;
    for (const auto& name : params) {
        env_.bind(name, param_tensors[idx++]);
    }

    torch::optim::SGD optimizer(param_tensors, torch::optim::SGDOptions(config.learningRate));

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        optimizer.zero_grad();
        forwardPass(program);

        if (!env_.has(rewardName)) {
            throw std::runtime_error("Reward tensor not found: " + rewardName);
        }

        torch::Tensor reward = env_.lookup(rewardName);
        if (reward.numel() > 1) {
            reward = reward.sum();
        }

        // Negate for maximization
        torch::Tensor loss = -reward;
        loss.backward();
        optimizer.step();

        if (config.verbose && (epoch % (config.epochs / 10) == 0 || epoch == config.epochs - 1)) {
            std::cout << "Epoch " << epoch << "/" << config.epochs
                      << " - Reward: " << reward.item<double>() << std::endl;
        }
    }

    return env_.lookup(rewardName).detach();
}

torch::Tensor LearningEngine::sample(
    const std::string& probName,
    const LearningConfig& config
) {
    if (!env_.has(probName)) {
        throw std::runtime_error("Probability tensor not found: " + probName);
    }

    torch::Tensor probs = env_.lookup(probName);

    // Ensure probabilities sum to 1
    probs = probs / probs.sum();

    // Sample from the distribution
    torch::Tensor samples = torch::multinomial(probs, config.sampleCount, true);

    return samples;
}

} // namespace tl
