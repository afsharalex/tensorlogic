// ============================================
// Conditional Data Preprocessing
// ============================================
// Apply different preprocessing steps based on data properties

// Raw sensor data with some outliers and missing values
RawData[0] = -999.0  // Missing value sentinel
RawData[1] = 10.0
RawData[2] = 15.0
RawData[3] = 200.0   // Outlier (too high)
RawData[4] = 12.0
RawData[5] = -999.0  // Missing value sentinel
RawData[6] = 8.0
RawData[7] = -50.0   // Outlier (negative)
RawData[8] = 11.0
RawData[9] = 14.0

// Preprocessing pipeline:
// - Replace missing values (-999) with mean (12.0)
// - Clip outliers (< 0 or > 100) to valid range
// - Keep valid values as-is

mean_value = 12.0
min_valid = 0.0
max_valid = 100.0

CleanData[i] = mean_value : (RawData[i] == -999.0)
             | min_valid : (RawData[i] < min_valid)
             | max_valid : (RawData[i] > max_valid)
             | RawData[i]

// Expected results:
// CleanData[0] = 12.0   (missing value replaced)
// CleanData[1] = 10.0   (valid)
// CleanData[2] = 15.0   (valid)
// CleanData[3] = 100.0  (outlier clipped)
// CleanData[4] = 12.0   (valid)
// CleanData[5] = 12.0   (missing value replaced)
// CleanData[6] = 8.0    (valid)
// CleanData[7] = 0.0    (outlier clipped)
// CleanData[8] = 11.0   (valid)
// CleanData[9] = 14.0   (valid)

CleanData[i]?