// ============================================
// File I/O with Multiple Files
// ============================================

// Load training features
X_train[i, j] = "Examples/Programs/data/train.csv"

// Load training labels
Y_train[i] = "Examples/Programs/data/labels.csv"

// Load test features
X_test[i, j] = "Examples/Programs/data/test.csv"

// Define weights (could also be loaded from file)
W[j] = "Examples/Programs/data/weights.csv"
bias = -0.5

// Make predictions on training data
train_predictions[i] = sigmoid(W[j] X_train[i, j] + bias)

// Make predictions on test data
test_predictions[i] = sigmoid(W[j] X_test[i, j] + bias)

// Write outputs to separate files
"Examples/Programs/output/train_predictions.txt" = train_predictions[i]
"Examples/Programs/output/test_predictions.txt" = test_predictions[i]

// Compute training accuracy (if we have labels)
// correct[i] = step(train_predictions[i] - 0.5) == Y_train[i]
// accuracy = correct[i] / 5.0

// Query
train_predictions[i]?
test_predictions[i]?