// ============================================
// Adaptive Pooling with Different Strides
// ============================================

// Input sequence of length 8
Sequence[0] = 1.0
Sequence[1] = 2.0
Sequence[2] = 3.0
Sequence[3] = 4.0
Sequence[4] = 5.0
Sequence[5] = 6.0
Sequence[6] = 7.0
Sequence[7] = 8.0

// Pool with stride 2 (4 outputs)
Pool_s2[i/2] avg= Sequence[i]
// Pool_s2[0] = avg(1.0, 2.0) = 1.5
// Pool_s2[1] = avg(3.0, 4.0) = 3.5
// Pool_s2[2] = avg(5.0, 6.0) = 5.5
// Pool_s2[3] = avg(7.0, 8.0) = 7.5

// Pool with stride 4 (2 outputs)
Pool_s4[i/4] avg= Sequence[i]
// Pool_s4[0] = avg(1.0, 2.0, 3.0, 4.0) = 2.5
// Pool_s4[1] = avg(5.0, 6.0, 7.0, 8.0) = 6.5

// Global pooling (1 output)
Global avg= Sequence[i]
// Global = avg(all 8 values) = 4.5

Pool_s2[i]?
Pool_s4[i]?
Global?