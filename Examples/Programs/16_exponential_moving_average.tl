// ============================================
// Exponential Moving Average with Virtual Index
// ============================================
// Update statistics online without storing full history

// Learning rate
alpha = 0.1

// Initial average
avg[0] = 0.0

// Streaming data points
data[0] = 5.0
data[1] = 8.0
data[2] = 6.0
data[3] = 9.0
data[4] = 7.0

// Exponential moving average update
// avg^{t+1} = (1 - α) * avg^t + α * data[t]
avg[*t+1] = (1.0 - alpha) * avg[*t] + alpha * data[t]

// This computes:
// t=0: avg[*1] = 0.9*0.0 + 0.1*5.0 = 0.5
// t=1: avg[*2] = 0.9*0.5 + 0.1*8.0 = 1.25
// t=2: avg[*3] = 0.9*1.25 + 0.1*6.0 = 1.725
// t=3: avg[*4] = 0.9*1.725 + 0.1*9.0 = 2.4525
// t=4: avg[*5] = 0.9*2.4525 + 0.1*7.0 = 2.90725

avg[0]?  // Final moving average