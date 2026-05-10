import pandas as pd
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.quality_detector import QualityDetector

# Create test data with quality issues
np.random.seed(42)

df = pd.DataFrame({
    'age': [25, 30, 45, 22, 35, 28, 40, 33, 29, 31],
    'income': [50000, 60000, 75000, 48000, 68000, 52000, 1000000, 55000, 62000, 58000],  # One outlier
    'city': ['NYC', 'LA', 'CHI', 'NYC', 'LA', 'NYC', 'CHI', 'LA', None, 'CHI'],  # One missing
    'price': [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000]  # One outlier + skewed
})

print("Test DataFrame:")
print(df)
print("\n" + "="*60)

detector = QualityDetector()

# Test 1: Missing Values
print("\n1️⃣ MISSING VALUES REPORT:")
missing = detector.detect_missing_values(df)
for col, stats in missing.items():
    print(f"   {col}: {stats['count']} missing ({stats['percentage']}%) → {stats['recommendation']}")

# Test 2: Outliers
print("\n2️⃣ OUTLIERS REPORT:")
outliers = detector.detect_outliers_iqr(df)
for col, stats in outliers.items():
    if stats['has_outliers']:
        print(f"   {col}: {stats['count']} outliers ({stats['percentage']}%)")
        print(f"      Bounds: [{stats['lower_bound']}, {stats['upper_bound']}]")

# Test 3: Skewness
print("\n3️⃣ SKEWNESS REPORT:")
skewness = detector.detect_skewness(df)
for col, stats in skewness.items():
    if stats['is_skewed']:
        print(f"   {col}: skewness={stats['skewness']} → {stats['recommendation']}")

# Test 4: Correlations
print("\n4️⃣ CORRELATIONS REPORT:")
correlations = detector.detect_correlations(df)
for col1, col2, corr in correlations:
    print(f"   {col1} ↔ {col2}: {corr}")

# Test 5: Full Report
print("\n" + "="*60)
full_report = detector.get_full_quality_report(df, verbose=True)