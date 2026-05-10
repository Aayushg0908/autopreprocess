"""
Test AutoClean pipeline on real Ames Housing dataset
"""

import pandas as pd
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import AutoClean

print("="*80)
print("🏠 TESTING ON REAL DATASET: AMES HOUSING")
print("="*80)

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('data/train.csv')
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Show dataset info
print(f"\n📋 Dataset info:")
print(f"   Target: SalePrice")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Preview columns and missing values
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
print(f"\n📊 Missing values: {len(missing_cols)} columns have missing data")
print(f"   Top 5 columns with missing:")
for col in missing_cols.nlargest(5).items():
    print(f"      {col[0]}: {col[1]} ({col[1]/len(df)*100:.1f}%)")

# Check target distribution
print(f"\n📊 Target distribution (SalePrice):")
print(f"   Min: ${df['SalePrice'].min():,.0f}")
print(f"   Max: ${df['SalePrice'].max():,.0f}")
print(f"   Mean: ${df['SalePrice'].mean():,.0f}")
print(f"   Median: ${df['SalePrice'].median():,.0f}")
print(f"   Skewness: {df['SalePrice'].skew():.2f}")

print("\n" + "="*80)
print("🤖 RUNNING AUTOCLEAN PIPELINE")
print("="*80)

start_time = time.time()

# Initialize pipeline with feature selection
pipeline = AutoClean(
    data=df,
    target='SalePrice',
    test_size=0.2,
    random_state=42,
    scaling=True,
    scaling_method='robust',  # Robust scaling handles outliers better
    encoding='auto',
    handle_missing=True,
    handle_outliers=True,
    remove_duplicates=True,
    drop_useless=True,
    feature_selection=True,
    selection_method='importance',
    top_k=20,  # Keep top 20 important features
    verbose=True
)

# Run preprocessing
X_train, X_test, y_train, y_test = pipeline.preprocess()

preprocess_time = time.time() - start_time

# Results
print("\n" + "="*80)
print("📊 PREPROCESSING RESULTS")
print("="*80)
print(f"   Time taken: {preprocess_time:.2f} seconds")
print(f"\n   Original data: {df.shape}")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   y_test: {y_test.shape}")

# Pipeline summary
pipeline.print_summary()

# Train a simple model
print("\n" + "="*80)
print("🤖 MODEL TRAINING & EVALUATION")
print("="*80)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Train Random Forest
print("\n📊 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n📈 Model Performance:")
print(f"\n   Training Set:")
print(f"      MAE:  ${train_mae:,.0f}")
print(f"      RMSE: ${train_rmse:,.0f}")
print(f"      R²:   {train_r2:.4f}")
print(f"\n   Test Set:")
print(f"      MAE:  ${test_mae:,.0f}")
print(f"      RMSE: ${test_rmse:,.0f}")
print(f"      R²:   {test_r2:.4f}")

# Feature importance
print("\n📊 Top 10 Most Important Features:")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False).head(10)

for i, row in feature_importance_df.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Save pipeline
print("\n" + "="*80)
print("💾 SAVING PIPELINE")
print("="*80)
pipeline.save('models/ames_pipeline.pkl')
print("✅ Pipeline saved to: models/ames_pipeline.pkl")

# Test predict_ready_data
print("\n" + "="*80)
print("🔮 TESTING predict_ready_data")
print("="*80)
X_new = X_test.head(5)
X_new_processed = pipeline.predict_ready_data(X_new)
print(f"   Input shape: {X_new.shape}")
print(f"   Output shape: {X_new_processed.shape}")
print("✅ predict_ready_data working!")

print("\n" + "="*80)
print("✅ AMES HOUSING TEST COMPLETE!")
print("="*80)

# Summary
print("\n📊 FINAL SUMMARY:")
print(f"   Original features: {df.shape[1] - 1} (excluding target)")
print(f"   Final features: {X_train.shape[1]}")
print(f"   Features removed: {(df.shape[1] - 1) - X_train.shape[1]}")
print(f"   Test R² score: {test_r2:.4f}")
print(f"   Test MAE: ${test_mae:,.0f}")
print(f"   Total time: {preprocess_time:.2f}s")