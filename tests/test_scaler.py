import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cleaner import DataCleaner
from core.encoder import Encoder
from core.scaler import Scaler

# Load dataset
print("="*70)
print("🧪 TESTING SCALER")
print("="*70)

df = pd.read_csv('data/test_dataset.csv')
print(f"\n📂 Loaded: {df.shape}")

# Clean data
cleaner = DataCleaner()
df_clean, _ = cleaner.remove_duplicates(df, verbose=False)
df_clean, _ = cleaner.drop_useless_columns(df_clean, verbose=False)
df_clean, _ = cleaner.handle_missing_values(df_clean, verbose=False)
df_clean, _ = cleaner.handle_outliers(df_clean, verbose=False)

# Encode categoricals
encoder = Encoder(verbose=False)
X_encoded = encoder.fit_transform(df_clean)

# Split into train/test (80/20)
train_size = int(0.8 * len(X_encoded))
X_train = X_encoded[:train_size]
X_test = X_encoded[train_size:]

print(f"\n📊 Train shape: {X_train.shape}")
print(f"📊 Test shape: {X_test.shape}")

# Test StandardScaler
print("\n" + "="*70)
print("TEST 1: StandardScaler")
print("="*70)
scaler_std = Scaler(method='standard', verbose=True)
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# Show before/after
numeric_cols = scaler_std.numeric_cols
print(f"\n📈 Before scaling (train):")
for col in numeric_cols[:3]:  # Show first 3
    print(f"   {col}: mean={X_train[col].mean():.2f}, std={X_train[col].std():.2f}")

print(f"\n📈 After scaling (train):")
for col in numeric_cols[:3]:
    print(f"   {col}: mean={X_train_std[col].mean():.2f}, std={X_train_std[col].std():.2f}")

# Test MinMaxScaler
print("\n" + "="*70)
print("TEST 2: MinMaxScaler")
print("="*70)
scaler_mm = Scaler(method='minmax', verbose=True)
X_train_mm = scaler_mm.fit_transform(X_train)

print(f"\n📈 Range after MinMax (train):")
for col in numeric_cols[:3]:
    print(f"   {col}: min={X_train_mm[col].min():.2f}, max={X_train_mm[col].max():.2f}")

# Test RobustScaler
print("\n" + "="*70)
print("TEST 3: RobustScaler")
print("="*70)
scaler_robust = Scaler(method='robust', verbose=True)
X_train_robust = scaler_robust.fit_transform(X_train)

print("\n✅ Scaler test complete!")