// ============================================
// File I/O Example
// ============================================
// This program demonstrates:
// 1. Reading data from CSV files
// 2. Processing the data (simple logistic regression)
// 3. Writing results to output files
//
// File I/O Syntax:
//   tensor = "filename"  -> reads file into tensor
//   "filename" = tensor  -> writes tensor to file

// ============================================
// Load Training Data from CSV
// ============================================
// Reading a CSV file creates a matrix where:
// - Rows (i) correspond to examples/samples
// - Columns (j) correspond to features
//
// The file "data/train.csv" contains:
// - 5 training examples
// - 3 features per example
// - Each row: feature1, feature2, feature3

X[i, j] = "Examples/Programs/data/train.csv"

// After loading, X is a 5x3 matrix:
// X[0, 0]=1.0, X[0, 1]=2.0, X[0, 2]=1.5
// X[1, 0]=2.0, X[1, 1]=3.0, X[1, 2]=2.5
// X[2, 0]=0.5, X[2, 1]=1.5, X[2, 2]=1.0
// X[3, 0]=3.0, X[3, 1]=4.0, X[3, 2]=3.5
// X[4, 0]=1.5, X[4, 1]=2.5, X[4, 2]=2.0

// ============================================
// Define Model Weights
// ============================================
// Weights for logistic regression
// Shape: [3] (one weight per feature)

W[0] = 0.5
W[1] = 0.3
W[2] = 0.4

// Bias term
bias = -1.0

// ============================================
// Process Data: Compute Predictions
// ============================================
// For each example i, compute:
// Y[i] = sigmoid(W[j] * X[i, j] + bias)
//
// This is a logistic regression model:
// - Computes weighted sum of features
// - Applies sigmoid to get probability (0 to 1)

// Compute weighted sum for each example
logits[i] = W[j] X[i, j] + bias

// Apply sigmoid activation
Y[i] = sigmoid(logits[i])

// Expected outputs (approximately):
// Y[0] = sigmoid(0.5*1.0 + 0.3*2.0 + 0.4*1.5 - 1.0)
//      = sigmoid(0.5 + 0.6 + 0.6 - 1.0)
//      = sigmoid(0.7)
//      ≈ 0.668
//
// Y[1] = sigmoid(0.5*2.0 + 0.3*3.0 + 0.4*2.5 - 1.0)
//      = sigmoid(1.0 + 0.9 + 1.0 - 1.0)
//      = sigmoid(1.9)
//      ≈ 0.870
//
// Y[2] = sigmoid(0.5*0.5 + 0.3*1.5 + 0.4*1.0 - 1.0)
//      = sigmoid(0.25 + 0.45 + 0.4 - 1.0)
//      = sigmoid(0.1)
//      ≈ 0.525
//
// Y[3] = sigmoid(0.5*3.0 + 0.3*4.0 + 0.4*3.5 - 1.0)
//      = sigmoid(1.5 + 1.2 + 1.4 - 1.0)
//      = sigmoid(3.1)
//      ≈ 0.957
//
// Y[4] = sigmoid(0.5*1.5 + 0.3*2.5 + 0.4*2.0 - 1.0)
//      = sigmoid(0.75 + 0.75 + 0.8 - 1.0)
//      = sigmoid(1.3)
//      ≈ 0.786

// ============================================
// Write Predictions to Output File
// ============================================
// Save the predictions to a text file
// Each line contains one prediction value

"Examples/Programs/output/predictions.txt" = Y[i]

// The output file will contain:
// 0.668
// 0.870
// 0.525
// 0.957
// 0.786

// ============================================
// Query Results (Before Writing)
// ============================================
// We can inspect the results before writing to file

Y[0]?  // Should return: ~0.668
Y[1]?  // Should return: ~0.870
Y[2]?  // Should return: ~0.525
Y[3]?  // Should return: ~0.957
Y[4]?  // Should return: ~0.786

// Query all predictions
Y[i]?  // Returns all 5 predictions