// Example 11: File I/O
// Reading from and writing to files

// Load data from CSV
X[i, j] = "data/train.csv"

// Process data
Y[i] = sigmoid(W[i, j] X[j])

// Write output to file
"output/predictions.txt" = Y[i]
