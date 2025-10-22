// ============================================
// Statistical Operations using Pooling
// ============================================

// Dataset: measurements across time
Data[0, 0] = 10.0
Data[0, 1] = 12.0
Data[0, 2] = 15.0
Data[0, 3] = 11.0
Data[0, 4] = 13.0
Data[1, 0] = 8.0
Data[1, 1] = 9.0
Data[1, 2] = 11.0
Data[1, 3] = 7.0
Data[1, 4] = 10.0

// Compute statistics for each sensor (row)

// Mean value across time
Mean[i] avg= Data[i, t]
// Mean[0] = (10+12+15+11+13)/5 = 12.2
// Mean[1] = (8+9+11+7+10)/5 = 9.0

// Maximum reading
Max[i] max= Data[i, t]
// Max[0] = 15.0
// Max[1] = 11.0

// Minimum reading
Min[i] min= Data[i, t]
// Min[0] = 10.0
// Min[1] = 7.0

// Range (requires multiple operations)
Range[i] = Max[i] - Min[i]
// Range[0] = 15.0 - 10.0 = 5.0
// Range[1] = 11.0 - 7.0 = 4.0

// Total (sum)
Total[i] += Data[i, t]
// Total[0] = 61.0
// Total[1] = 45.0

Mean[i]?
Max[i]?
Min[i]?
Range[i]?
Total[i]?