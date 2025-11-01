#pragma once

#include "TL/AST.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_set>

namespace tl {

// Forward declarations
class Environment;
class TensorBackend;
class ExecutorRegistry;

// Learning configuration extracted from directive arguments
struct LearningConfig {
    std::string directive;  // "minimize", "maximize", "sample"
    double learningRate{0.01};
    int epochs{100};
    int sampleCount{1000};
    bool verbose{false};

    // Parse from QueryDirective
    static LearningConfig fromDirective(const QueryDirective& dir);
};

// The LearningEngine handles gradient-based learning using PyTorch autograd
class LearningEngine {
public:
    explicit LearningEngine(Environment& env, TensorBackend& backend, ExecutorRegistry& registry, std::ostream* output = nullptr);

    // Execute a learning directive on a target tensor
    // Returns the final value of the target after optimization
    torch::Tensor executeDirective(
        const std::string& targetName,
        const QueryDirective& directive,
        const Program& program
    );

    // Minimize a loss tensor by adjusting learnable parameters
    torch::Tensor minimize(
        const std::string& lossName,
        const LearningConfig& config,
        const Program& program
    );

    // Maximize a reward tensor (internally minimizes negative)
    torch::Tensor maximize(
        const std::string& rewardName,
        const LearningConfig& config,
        const Program& program
    );

    // Sample from a probability distribution
    torch::Tensor sample(
        const std::string& probName,
        const LearningConfig& config
    );

private:
    Environment& env_;
    TensorBackend& backend_;
    ExecutorRegistry& registry_;
    std::ostream* output_;

    // Identify learnable parameters (tensors not derived from other tensors)
    std::unordered_set<std::string> identifyLearnableParameters(const Program& program);

    // Mark tensors as requiring gradients
    void enableGradients(const std::unordered_set<std::string>& params);

    // Perform one forward pass through the program
    void forwardPass(const Program& program);

    // Perform backward pass and update parameters
    void backwardPass(
        const std::string& targetName,
        const std::unordered_set<std::string>& params,
        torch::optim::Optimizer& optimizer
    );

    // Zero gradients for all parameters
    void zeroGradients(const std::unordered_set<std::string>& params);
};

} // namespace tl
