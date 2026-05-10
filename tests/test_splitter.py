import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.splitter import DataSplitter

# Load dataset
print("="*70)
print("🧪 TESTING SPLITTER")
print("="*70)

df = pd.read_csv('data/test_dataset.csv')
print(f"\n📂 Loaded: {df.shape}")

# Use 'rating' as target for classification example
# Use 'total_spent' for regression example

# Test 1: Random split (default)
print("\n" + "="*70)
print("TEST 1: Random Split (80/20)")
print("="*70)
splitter = DataSplitter(test_size=0.2, random_state=42, verbose=True)
X_train, X_test, y_train, y_test = splitter.split_dataframe(
    df, target_col='rating', strategy='random'
)

print(f"\n✅ Random split results:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   y_test: {y_test.shape}")

# Test 2: Stratified split
print("\n" + "="*70)
print("TEST 2: Stratified Split (preserve class balance)")
print("="*70)
splitter2 = DataSplitter(test_size=0.3, random_state=42, verbose=True)
X_train2, X_test2, y_train2, y_test2 = splitter2.split_dataframe(
    df, target_col='rating', strategy='stratified'
)

# Test 3: Time series split
print("\n" + "="*70)
print("TEST 3: Time Series Split (no shuffle)")
print("="*70)
splitter3 = DataSplitter(test_size=0.2, random_state=42, verbose=True)
X_train3, X_test3, y_train3, y_test3 = splitter3.split_dataframe(
    df, target_col='total_spent', strategy='time_series'
)

# Test 4: Custom train size
print("\n" + "="*70)
print("TEST 4: Custom Train Size (75/25)")
print("="*70)
splitter4 = DataSplitter(train_size=0.75, random_state=42, verbose=True)
X_train4, X_test4, y_train4, y_test4 = splitter4.split_dataframe(
    df, target_col='rating', strategy='random'
)

print(f"\n✅ Custom split: train={len(X_train4)}, test={len(X_test4)}")

print("\n✅ Splitter test complete!")