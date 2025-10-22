// ============================================
// 2D Pooling for Image Processing
// ============================================
// Common in convolutional neural networks
// Pool over spatial dimensions with stride

// Define a 4x4 image (single channel)
// Shape: [4, 4]
//
//        y=0  y=1  y=2  y=3
// x=0  [ 1.0  2.0  3.0  4.0 ]
// x=1  [ 5.0  6.0  7.0  8.0 ]
// x=2  [ 9.0  8.0  7.0  6.0 ]
// x=3  [ 5.0  4.0  3.0  2.0 ]

Image[0, 0] = 1.0
Image[0, 1] = 2.0
Image[0, 2] = 3.0
Image[0, 3] = 4.0
Image[1, 0] = 5.0
Image[1, 1] = 6.0
Image[1, 2] = 7.0
Image[1, 3] = 8.0
Image[2, 0] = 9.0
Image[2, 1] = 8.0
Image[2, 2] = 7.0
Image[2, 3] = 6.0
Image[3, 0] = 5.0
Image[3, 1] = 4.0
Image[3, 2] = 3.0
Image[3, 3] = 2.0

// 2x2 Max Pooling with stride 2
// Reduces 4x4 to 2x2 by taking max in each 2x2 block
//
// Block (0,0): max(1.0, 2.0, 5.0, 6.0) = 6.0
// Block (0,1): max(3.0, 4.0, 7.0, 8.0) = 8.0
// Block (1,0): max(9.0, 8.0, 5.0, 4.0) = 9.0
// Block (1,1): max(7.0, 6.0, 3.0, 2.0) = 7.0

// Using integer division for pooling with stride 2
Pooled_max[x/2, y/2] max= Image[x, y]

// Result will be:
// Pooled_max[0, 0] = 6.0
// Pooled_max[0, 1] = 8.0
// Pooled_max[1, 0] = 9.0
// Pooled_max[1, 1] = 7.0

// 2x2 Average Pooling with stride 2
Pooled_avg[x/2, y/2] avg= Image[x, y]

// Result will be:
// Pooled_avg[0, 0] = (1.0 + 2.0 + 5.0 + 6.0) / 4 = 3.5
// Pooled_avg[0, 1] = (3.0 + 4.0 + 7.0 + 8.0) / 4 = 5.5
// Pooled_avg[1, 0] = (9.0 + 8.0 + 5.0 + 4.0) / 4 = 6.5
// Pooled_avg[1, 1] = (7.0 + 6.0 + 3.0 + 2.0) / 4 = 4.5

Pooled_max[0, 0]?  // 6.0
Pooled_max[0, 1]?  // 8.0
Pooled_avg[0, 0]?  // 3.5
Pooled_avg[1, 1]?  // 4.5