import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.type_detector import TypeDetector

# Create test data
df = pd.DataFrame({
    'age': [25, 30, 45, 22, 35, 28, 40, 33, 29, 55],           # Many unique → continuous
    'rating': [1, 2, 3, 2, 1, 3, 2, 1, 3, 2],                 # Few unique, small range → discrete_numeric
    'education_level': [1, 2, 3, 2, 1, 3, 2, 1, 3, 2],        # 1=HS, 2=College, 3=Grad → discrete_numeric
    'city': ['NYC', 'LA', 'CHI', 'NYC', 'LA', 'NYC', 'CHI', 'LA', 'NYC', 'CHI'],  # Categorical
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', 
             '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', 
             '2024-01-09', '2024-01-10'],                      # Datetime
    'customer_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],  # ID → useless
    'constant': ['X'] * 10,                                     # Constant → useless
    'price': [19.99, 29.99, 39.99, 24.99, 34.99, 44.99, 29.99, 19.99, 49.99, 39.99]  # Many unique → continuous
})

print("="*60)
print("DATA TYPES DETECTION TEST")
print("="*60)

detector = TypeDetector()
analysis = detector.analyze_dataframe(df, verbose=True)

print("\n" + "="*60)
print("CLASSIFICATION RESULTS:")
print("="*60)
print(f"Continuous (scale): {analysis['continuous']}")
print(f"Discrete numeric (keep as numbers): {analysis['discrete_numeric']}")
print(f"Categorical (encode): {analysis['categorical']}")
print(f"Datetime: {analysis['datetime']}")
print(f"Useless (drop): {analysis['useless']}")

print("\n" + "="*60)
print("COLUMN SUMMARY:")
print("="*60)
summary = detector.get_column_summary(df)
print(summary.to_string(index=False))