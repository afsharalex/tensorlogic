// ============================================
// Time-Series Conditional Processing
// ============================================
// Apply different filters based on time-of-day

// Sensor readings at different hours
Hour[0] = 2.0   // Night
Hour[1] = 8.0   // Morning rush
Hour[2] = 14.0  // Afternoon
Hour[3] = 18.0  // Evening rush
Hour[4] = 22.0  // Night

Readings[0] = 10.0
Readings[1] = 50.0
Readings[2] = 30.0
Readings[3] = 60.0
Readings[4] = 15.0

// Apply different smoothing based on time
// - Night (0-6, 22-24): light smoothing (0.9)
// - Rush hours (7-9, 17-19): heavy smoothing (0.5)
// - Normal hours: moderate smoothing (0.7)

Smoothed[i] = 0.9 * Readings[i] : (Hour[i] >= 0.0 and Hour[i] < 6.0)
            | 0.9 * Readings[i] : (Hour[i] >= 22.0)
            | 0.5 * Readings[i] : (Hour[i] >= 7.0 and Hour[i] <= 9.0)
            | 0.5 * Readings[i] : (Hour[i] >= 17.0 and Hour[i] <= 19.0)
            | 0.7 * Readings[i]

// Expected results:
// Smoothed[0] = 0.9 * 10 = 9.0   (night)
// Smoothed[1] = 0.5 * 50 = 25.0  (morning rush)
// Smoothed[2] = 0.7 * 30 = 21.0  (afternoon)
// Smoothed[3] = 0.5 * 60 = 30.0  (evening rush)
// Smoothed[4] = 0.9 * 15 = 13.5  (night)

Smoothed[i]?