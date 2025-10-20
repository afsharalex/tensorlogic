// ============================================
// Additional Modern Activation Functions
// ============================================

input = [-2.0, -1.0, 0.0, 1.0, 2.0]

// Leaky ReLU: allows small negative gradient
// LeakyReLU(x) = max(0.01*x, x)
alpha = 0.01
leaky_relu[i] = relu(input[i]) + alpha * relu(-input[i])
// Expected: [-0.02, -0.01, 0.0, 1.0, 2.0]

// ELU (Exponential Linear Unit): smooth negative part
// ELU(x) = x if x>0, else alpha*(e^x - 1)
elu[i] = relu(input[i]) + alpha * (step(-input[i]) * (exp(input[i]) - 1.0))

// Swish (also called SiLU): smooth, self-gated
// Swish(x) = x * sigmoid(x)
swish[i] = input[i] * sigmoid(input[i])
// Expected: [-0.238, -0.268, 0.0, 0.731, 1.762]

leaky_relu?
elu?
swish?