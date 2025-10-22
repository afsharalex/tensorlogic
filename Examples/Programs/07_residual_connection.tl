// ============================================
// Residual Connection (Skip Connection)
// ============================================
// Used in ResNets: output = F(x) + x
// Helps with gradient flow in deep networks

// Input
x[0] = 1.0
x[1] = 2.0
x[2] = 3.0

// Apply transformation F(x) - e.g., weight matrix
W[0] = 0.3
W[1] = 0.5
W[2] = 0.2

transformed = W[i] * x[i]
// transformed = 0.3*1.0 + 0.5*2.0 + 0.2*3.0
//             = 0.3 + 1.0 + 0.6 = 1.9

// Add residual connection
// output = F(x) + x (element-wise for each component)
// For scalars: output = transformed + (x[0] + x[1] + x[2])
input_sum = x[i]  // 6.0
output = transformed + input_sum  // 1.9 + 6.0 = 7.9

transformed?  // 1.9
input_sum?    // 6.0
output?       // 7.9