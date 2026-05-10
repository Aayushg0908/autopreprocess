"""
Test encoder with frequency encoding, one-hot encoding, and datetime extraction
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cleaner import DataCleaner
from core.encoder import Encoder

# Load dataset
print("="*70)
print("🧪 TESTING ENCODER")
print("="*70)

df = pd.read_csv('data/test_dataset.csv')
print(f"\n📂 Loaded: {df.shape}")

# First clean the data
cleaner = DataCleaner()
df_clean, _ = cleaner.remove_duplicates(df, verbose=False)
df_clean, _ = cleaner.drop_useless_columns(df_clean, verbose=False)
df_clean, _ = cleaner.handle_missing_values(df_clean, verbose=False)
df_clean, _ = cleaner.handle_outliers(df_clean, verbose=False)

print(f"\n🧹 After cleaning: {df_clean.shape}")

# Separate features (X) - exclude customer_id and constant_col already dropped
# Keep all columns for encoding test
X = df_clean.copy()

# Initialize encoder
encoder = Encoder(verbose=True)

# Fit and transform
X_encoded = encoder.fit_transform(X)

print("\n" + "="*70)
print("📊 ENCODING RESULTS")
print("="*70)
print(f"\nOriginal shape: {X.shape}")
print(f"Encoded shape: {X_encoded.shape}")
print(f"\nFirst 5 rows of encoded data:")
print(X_encoded.head())

print("\n✅ Encoder test complete!")