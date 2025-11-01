// Implementing Activation Functions with Comparisons
// Shows how ReLU and other activations can be implemented using comparison operators

// Input data with positive and negative values
Input[0] = -5.0
Input[1] = -2.5
Input[2] = -0.1
Input[3] = 0.0
Input[4] = 0.1
Input[5] = 2.5
Input[6] = 5.0

// 1. Manual ReLU implementation using comparison
// ReLU(x) = max(0, x) = x * (x > 0)
manual_relu[i] = Input[i] * (Input[i] > 0.0)

// 2. Leaky ReLU: allows small negative slope
// LeakyReLU(x) = x if x > 0, else alpha * x
alpha = 0.1
positive_mask[i] = Input[i] > 0.0
negative_mask[i] = Input[i] <= 0.0
leaky_relu[i] = (Input[i] * positive_mask[i]) + (alpha * Input[i] * negative_mask[i])

// 3. Hard Tanh: clips values to [-1, 1]
// HardTanh(x) = -1 if x < -1, 1 if x > 1, else x
below_neg_one[i] = Input[i] < -1.0
above_pos_one[i] = Input[i] > 1.0
in_range[i] = (Input[i] >= -1.0) and (Input[i] <= 1.0)
hard_tanh[i] = (-1.0 * below_neg_one[i]) + (1.0 * above_pos_one[i]) + (Input[i] * in_range[i])

// 4. Threshold activation: binary step function
// Step(x) = 1 if x > threshold, else 0
threshold = 0.5
step_activation[i] = Input[i] > threshold

// 5. Clipped Linear Unit (CLU): clips to [min, max]
min_clip = -1.0
max_clip = 3.0
too_low[i] = Input[i] < min_clip
too_high[i] = Input[i] > max_clip
normal[i] = (Input[i] >= min_clip) and (Input[i] <= max_clip)
clu[i] = (min_clip * too_low[i]) + (max_clip * too_high[i]) + (Input[i] * normal[i])

// 6. Absolute value using comparison
// abs(x) = x if x >= 0, else -x
abs_input[i] = (Input[i] * positive_mask[i]) + (-1.0 * Input[i] * negative_mask[i])
