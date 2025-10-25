// ============================================
// Custom Activation with Guards
// ============================================
// Define a custom activation function with different behaviors
// in different ranges

// Input activations
Activations[0] = -10.0
Activations[1] = -2.0
Activations[2] = -0.5
Activations[3] = 0.0
Activations[4] = 0.5
Activations[5] = 2.0
Activations[6] = 10.0

// Custom activation:
// - Clip very negative values to 0 (dead neurons)
// - Linear in small range around 0
// - ReLU for moderate positive
// - Saturate at high values

Output[i] = 0.0 : (Activations[i] < -5.0)
          | Activations[i] + 5.0 : (Activations[i] >= -5.0 and Activations[i] < -1.0)
          | Activations[i] : (Activations[i] >= -1.0 and Activations[i] <= 1.0)
          | relu(Activations[i]) : (Activations[i] > 1.0 and Activations[i] < 5.0)
          | 5.0

// Expected results:
// Output[0] = 0.0          (x < -5, clipped)
// Output[1] = -2 + 5 = 3.0 (in linear range)
// Output[2] = -0.5         (identity range)
// Output[3] = 0.0          (identity range)
// Output[4] = 0.5          (identity range)
// Output[5] = relu(2) = 2.0 (relu range)
// Output[6] = 5.0          (saturated)

Output[i]?