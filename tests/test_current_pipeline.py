"""
Test only the completed components with method checking
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.memory import reduce_memory_usage
from detectors.type_detector import TypeDetector
from detectors.quality_detector import QualityDetector
from core.cleaner import DataCleaner

print("="*70)
print("🧪 TESTING COMPLETED COMPONENTS")
print("="*70)

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('data/test_dataset.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Test 1: Memory Optimization
print("\n" + "="*70)
print("TEST 1: MEMORY OPTIMIZATION (utils/memory.py)")
print("="*70)
df_optimized = reduce_memory_usage(df, verbose=True)

# Test 2: Type Detection
print("\n" + "="*70)
print("TEST 2: TYPE DETECTION (detectors/type_detector.py)")
print("="*70)
detector = TypeDetector()
type_analysis = detector.analyze_dataframe(df_optimized, verbose=True)

print("\n📊 Type Summary:")
summary = detector.get_column_summary(df_optimized)
print(summary.to_string(index=False))

# Test 3: Quality Detection
print("\n" + "="*70)
print("TEST 3: QUALITY DETECTION (detectors/quality_detector.py)")
print("="*70)
quality_report = QualityDetector.get_full_quality_report(df_optimized, verbose=True)

# Test 4: Data Cleaning
print("\n" + "="*70)
print("TEST 4: DATA CLEANING (core/cleaner.py)")
print("="*70)

cleaner = DataCleaner()

# First, let's see what methods are available
print("\n📋 Available methods in DataCleaner:")
available_methods = [method for method in dir(cleaner) if not method.startswith('_')]
for method in available_methods:
    print(f"   - {method}")

df_cleaned = df_optimized.copy()

# Try each method if it exists
if 'remove_duplicates' in available_methods:
    print("\n✅ Running remove_duplicates...")
    result = cleaner.remove_duplicates(df_cleaned, verbose=True)
    if isinstance(result, tuple) and len(result) == 2:
        df_cleaned, dup_removed = result
    else:
        df_cleaned = result

if 'drop_useless_columns' in available_methods:
    print("\n✅ Running drop_useless_columns...")
    result = cleaner.drop_useless_columns(df_cleaned, verbose=True)
    if isinstance(result, tuple) and len(result) == 2:
        df_cleaned, dropped = result
    else:
        df_cleaned = result
else:
    print("\n⚠️ drop_useless_columns not found - skipping")

if 'handle_missing_values' in available_methods:
    print("\n✅ Running handle_missing_values...")
    result = cleaner.handle_missing_values(df_cleaned, verbose=True)
    if isinstance(result, tuple) and len(result) == 2:
        df_cleaned, actions = result
    else:
        df_cleaned = result
else:
    print("\n⚠️ handle_missing_values not found - skipping")

if 'handle_outliers' in available_methods:
    print("\n✅ Running handle_outliers...")
    result = cleaner.handle_outliers(df_cleaned, verbose=True)
    if isinstance(result, tuple) and len(result) == 2:
        df_cleaned, info = result
    else:
        df_cleaned = result
else:
    print("\n⚠️ handle_outliers not found - skipping")

# Final Summary
print("\n" + "="*70)
print("✅ TEST COMPLETE")
print("="*70)
print(f"\n📈 Results Summary:")
print(f"   Original shape: {df.shape}")
print(f"   Final shape: {df_cleaned.shape}")
print(f"   Quality score: {quality_report['quality_score']}/100")