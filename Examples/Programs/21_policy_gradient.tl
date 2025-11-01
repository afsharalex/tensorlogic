// Policy Gradient Example
// Simple bandit problem with 3 arms
// Learn policy that maximizes expected reward

// Policy parameters (unnormalized action preferences)
Theta = [0.0, 0.0, 0.0]

// Softmax policy: convert preferences to probabilities
// π(a) = exp(θ_a) / Σ exp(θ_i)
Exp_theta[a] = exp(Theta[a])
Sum_exp = Exp_theta[a]
Policy[a] = Exp_theta[a] / Sum_exp

// Simulated rewards for each action (true values unknown to agent)
True_rewards = [1.0, 5.0, 3.0]

// Expected reward under current policy
Expected_reward = Policy[a] True_rewards[a]

// Query initial policy
Policy?

// Maximize expected reward
Expected_reward? @maximize(lr=0.1, epochs=100, verbose=true)

// Query learned policy
Policy?

// Query learned preferences
Theta?
