// ============================================
// Advanced Slicing: Practical Applications
// ============================================
// This program demonstrates real-world use cases for tensor slicing
// in neural networks and data processing pipelines

// ============================================
// Use Case 1: Feature Selection
// ============================================
// In many ML tasks, you need to select specific features from a dataset
// Example: Extract specific columns from a feature matrix

// Define feature matrix with 4 samples, 6 features each
// Features represent: [age, income, score1, score2, score3, label]
// Shape: [4, 6]
//
// Visual representation:
//          age  income  sc1   sc2   sc3   label
// sample0 [ 25    50    0.8   0.7   0.9    1.0 ]
// sample1 [ 30    60    0.6   0.8   0.7    0.0 ]
// sample2 [ 35    70    0.9   0.6   0.8    1.0 ]
// sample3 [ 40    80    0.7   0.9   0.6    0.0 ]

Features[0, 0] = 25.0
Features[0, 1] = 50.0
Features[0, 2] = 0.8
Features[0, 3] = 0.7
Features[0, 4] = 0.9
Features[0, 5] = 1.0

Features[1, 0] = 30.0
Features[1, 1] = 60.0
Features[1, 2] = 0.6
Features[1, 3] = 0.8
Features[1, 4] = 0.7
Features[1, 5] = 0.0

Features[2, 0] = 35.0
Features[2, 1] = 70.0
Features[2, 2] = 0.9
Features[2, 3] = 0.6
Features[2, 4] = 0.8
Features[2, 5] = 1.0

Features[3, 0] = 40.0
Features[3, 1] = 80.0
Features[3, 2] = 0.7
Features[3, 3] = 0.9
Features[3, 4] = 0.6
Features[3, 5] = 0.0

// Extract only the score features (columns 2, 3, 4)
// This is common when you want to use only certain features
// Scores shape: [4, 3]
Scores = Features[:, 2:5]

// Extract the labels (last column)
// Labels shape: [4]
Labels = Features[:, 5]

// Query feature selection results
Scores[0, 0]?  // Should return: 0.8 (score1 of sample0)
Scores[1, 2]?  // Should return: 0.7 (score3 of sample1)
Labels[0]?     // Should return: 1.0
Labels[1]?     // Should return: 0.0

// ============================================
// Use Case 2: Train/Test Split
// ============================================
// Split data into training and testing sets
// Common ratio: 75% train, 25% test
// With 4 samples: first 3 for training, last 1 for testing

// Training data (first 3 samples, all features)
// TrainX shape: [3, 6]
TrainX = Features[:3, :]

// Test data (last sample, all features)
// TestX shape: [1, 6]
TestX = Features[3:, :]

// Query split results
TrainX[0, 0]?  // Should return: 25.0 (age of first training sample)
TrainX[2, 5]?  // Should return: 1.0 (label of last training sample)
TestX[0, 0]?   // Should return: 40.0 (age of test sample)

// ============================================
// Use Case 3: Time Series Window Extraction
// ============================================
// Extract sliding windows from time series data
// Common in sequence modeling and forecasting

// Define time series (12 time steps)
TimeSeries[0] = 1.0
TimeSeries[1] = 2.0
TimeSeries[2] = 3.0
TimeSeries[3] = 4.0
TimeSeries[4] = 5.0
TimeSeries[5] = 6.0
TimeSeries[6] = 7.0
TimeSeries[7] = 8.0
TimeSeries[8] = 9.0
TimeSeries[9] = 10.0
TimeSeries[10] = 11.0
TimeSeries[11] = 12.0

// Extract first window (time steps 0-4)
// Window1 = [1, 2, 3, 4, 5]
Window1 = TimeSeries[0:5]

// Extract second window (time steps 3-7)
// Window2 = [4, 5, 6, 7, 8]
Window2 = TimeSeries[3:8]

// Extract last window (time steps 7-11)
// Window3 = [8, 9, 10, 11, 12]
Window3 = TimeSeries[7:12]

// Query window results
Window1[0]?  // Should return: 1.0
Window1[4]?  // Should return: 5.0
Window2[0]?  // Should return: 4.0
Window3[4]?  // Should return: 12.0

// ============================================
// Use Case 4: Batch Processing with Stride
// ============================================
// Process data in batches with a specific stride
// Useful for mini-batch gradient descent

// Define dataset with 10 samples
Dataset[0] = 0.0
Dataset[1] = 1.0
Dataset[2] = 2.0
Dataset[3] = 3.0
Dataset[4] = 4.0
Dataset[5] = 5.0
Dataset[6] = 6.0
Dataset[7] = 7.0
Dataset[8] = 8.0
Dataset[9] = 9.0

// Extract every other sample (batch size 2, stride 2)
// Batch1 = [0, 2, 4, 6, 8]
Batch1 = Dataset[::2]

// Extract offset samples
// Batch2 = [1, 3, 5, 7, 9]
Batch2 = Dataset[1::2]

// Query batch results
Batch1[0]?  // Should return: 0.0
Batch1[2]?  // Should return: 4.0
Batch2[0]?  // Should return: 1.0
Batch2[4]?  // Should return: 9.0

// ============================================
// Use Case 5: Image Patch Extraction
// ============================================
// Extract patches from a simple "image" (2D tensor)
// Common in convolutional neural networks and vision tasks

// Define a 6x6 "image" with different regions
Image[0, 0] = 1.0
Image[0, 1] = 1.0
Image[0, 2] = 2.0
Image[0, 3] = 2.0
Image[0, 4] = 3.0
Image[0, 5] = 3.0

Image[1, 0] = 1.0
Image[1, 1] = 1.0
Image[1, 2] = 2.0
Image[1, 3] = 2.0
Image[1, 4] = 3.0
Image[1, 5] = 3.0

Image[2, 0] = 4.0
Image[2, 1] = 4.0
Image[2, 2] = 5.0
Image[2, 3] = 5.0
Image[2, 4] = 6.0
Image[2, 5] = 6.0

Image[3, 0] = 4.0
Image[3, 1] = 4.0
Image[3, 2] = 5.0
Image[3, 3] = 5.0
Image[3, 4] = 6.0
Image[3, 5] = 6.0

Image[4, 0] = 7.0
Image[4, 1] = 7.0
Image[4, 2] = 8.0
Image[4, 3] = 8.0
Image[4, 4] = 9.0
Image[4, 5] = 9.0

Image[5, 0] = 7.0
Image[5, 1] = 7.0
Image[5, 2] = 8.0
Image[5, 3] = 8.0
Image[5, 4] = 9.0
Image[5, 5] = 9.0

// Extract top-left 2x2 patch
// TopLeftPatch:
// [ 1.0  1.0 ]
// [ 1.0  1.0 ]
TopLeftPatch = Image[:2, :2]

// Extract center 2x2 patch
// CenterPatch:
// [ 5.0  5.0 ]
// [ 5.0  5.0 ]
CenterPatch = Image[2:4, 2:4]

// Extract bottom-right 2x2 patch
// BottomRightPatch:
// [ 9.0  9.0 ]
// [ 9.0  9.0 ]
BottomRightPatch = Image[4:6, 4:6]

// Query patch results
TopLeftPatch[0, 0]?     // Should return: 1.0
CenterPatch[0, 0]?      // Should return: 5.0
BottomRightPatch[1, 1]? // Should return: 9.0

// ============================================
// Use Case 6: Attention Mechanism Context Window
// ============================================
// Extract context windows for attention computation
// Used in transformers and sequence-to-sequence models

// Define sequence embeddings (8 time steps, 3 dimensions each)
// For simplicity, use position indices as values
Embeddings[0, 0] = 0.0
Embeddings[0, 1] = 0.1
Embeddings[0, 2] = 0.2
Embeddings[1, 0] = 1.0
Embeddings[1, 1] = 1.1
Embeddings[1, 2] = 1.2
Embeddings[2, 0] = 2.0
Embeddings[2, 1] = 2.1
Embeddings[2, 2] = 2.2
Embeddings[3, 0] = 3.0
Embeddings[3, 1] = 3.1
Embeddings[3, 2] = 3.2
Embeddings[4, 0] = 4.0
Embeddings[4, 1] = 4.1
Embeddings[4, 2] = 4.2
Embeddings[5, 0] = 5.0
Embeddings[5, 1] = 5.1
Embeddings[5, 2] = 5.2
Embeddings[6, 0] = 6.0
Embeddings[6, 1] = 6.1
Embeddings[6, 2] = 6.2
Embeddings[7, 0] = 7.0
Embeddings[7, 1] = 7.1
Embeddings[7, 2] = 7.2

// Extract context for position 3 (window size 3: positions 2, 3, 4)
// Context3 shape: [3, 3]
Context3 = Embeddings[2:5, :]

// Extract context for position 5 (window size 3: positions 4, 5, 6)
// Context5 shape: [3, 3]
Context5 = Embeddings[4:7, :]

// Query context results
Context3[0, 0]?  // Should return: 2.0 (position 2, dim 0)
Context3[1, 1]?  // Should return: 3.1 (position 3, dim 1)
Context3[2, 2]?  // Should return: 4.2 (position 4, dim 2)

Context5[0, 0]?  // Should return: 4.0 (position 4, dim 0)
Context5[2, 2]?  // Should return: 6.2 (position 6, dim 2)
