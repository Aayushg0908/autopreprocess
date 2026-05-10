import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.selector import FeatureSelector
from core.pipeline import AutoClean

# Load and preprocess data
print("="*70)
print("🧪 TESTING FEATURE SELECTOR")
print("="*70)

# First get preprocessed data
pipeline = AutoClean(
    data='data/test_dataset.csv',
    target='rating',
    verbose=False
)
X_train, X_test, y_train, y_test = pipeline.preprocess()

print(f"\n📂 Preprocessed data shape: {X_train.shape}")

# Test 1: Variance Threshold
print("\n" + "="*70)
print("TEST 1: Variance Threshold")
print("="*70)
selector1 = FeatureSelector(method='variance', variance_threshold=0.01, verbose=True)
X_selected1 = selector1.fit_transform(X_train, y_train)
print(f"\n✅ Result: {X_train.shape[1]} → {X_selected1.shape[1]} features")
selector1.print_summary()

# Test 2: Correlation Removal
print("\n" + "="*70)
print("TEST 2: Correlation Removal")
print("="*70)
selector2 = FeatureSelector(method='correlation', correlation_threshold=0.8, verbose=True)
X_selected2 = selector2.fit_transform(X_train, y_train)
print(f"\n✅ Result: {X_train.shape[1]} → {X_selected2.shape[1]} features")
selector2.print_summary()

# Test 3: Feature Importance (Top 10)
print("\n" + "="*70)
print("TEST 3: Feature Importance (Top 10)")
print("="*70)
selector3 = FeatureSelector(method='importance', top_k=10, verbose=True)
X_selected3 = selector3.fit_transform(X_train, y_train)
print(f"\n✅ Result: {X_train.shape[1]} → {X_selected3.shape[1]} features")
print("\n📈 Feature Importances:")
for feat, imp in list(selector3.feature_importances_.items())[:5]:
    print(f"   {feat}: {imp:.4f}")
selector3.print_summary()

# Test 4: Mutual Information (Top 15)
print("\n" + "="*70)
print("TEST 4: Mutual Information (Top 15)")
print("="*70)
selector4 = FeatureSelector(method='mutual_info', top_k=15, verbose=True)
X_selected4 = selector4.fit_transform(X_train, y_train)
print(f"\n✅ Result: {X_train.shape[1]} → {X_selected4.shape[1]} features")
selector4.print_summary()

print("\n✅ Feature selector test complete!")