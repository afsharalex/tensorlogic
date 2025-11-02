// Linear Regression Example
// Learn parameters m (slope) and b (intercept) to fit y = mx + b
// Given training data points

// Training data: y = 2x + 1 with some noise
X_data = [1.0, 2.0, 3.0, 4.0, 5.0]
Y_data = [3.1, 4.9, 7.2, 9.0, 10.8]

// Learnable parameters (initialized randomly)
m = [0.5]
b = [0.0]

// Model: predictions = m * x + b
Predictions[i] = m[0] X_data[i] + b[0]

// Loss: Mean Squared Error
Errors[i] = (Predictions[i] - Y_data[i])^2
Loss = Errors[i]  // Sum over i

// Query the initial loss
Loss?

// Optimize parameters to minimize loss
Loss? @minimize(lr=0.01, epochs=100, verbose=true)

// Query learned parameters
m?
b?
