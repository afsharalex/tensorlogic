// Sampling from Probability Distributions
// Demonstrates the @sample directive

// Example 1: Categorical distribution
// Define probabilities for 5 outcomes (will be normalized)
Probs = [0.5, 1.0, 2.0, 1.5, 1.0]

// Query the distribution
Probs?

// Sample 1000 outcomes from this distribution
Probs? @sample(n=1000)

// Example 2: Learned distribution
// Start with uniform and shape it via optimization
Uniform = [1.0, 1.0, 1.0, 1.0]

// Target distribution we want to match
Target = [0.1, 0.2, 0.4, 0.3]

// Normalize to get probabilities
Uniform_norm[i] = Uniform[i] / Uniform[j]
Target_norm[i] = Target[i] / Target[j]

// KL divergence loss: D_KL(Target || Uniform)
// KL(P||Q) = Σ p(i) log(p(i)/q(i))
Log_ratio[i] = log(Target_norm[i] / Uniform_norm[i])
KL_loss = Target_norm[i] Log_ratio[i]

// Minimize KL divergence to match target
KL_loss? @minimize(lr=0.05, epochs=100, verbose=true)

// Sample from learned distribution
Uniform? @sample(n=1000)
