import pandas as pd
from autopreprocess import AutoClean

print("="*70)
print("🏠 TESTING ON AMES HOUSING DATASET")
print("="*70)

# Try loading the dataset
try:
    df = pd.read_csv('data/train.csv')
    print(f"\n✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    # Use California Housing as fallback
    print("\n⚠️ Ames dataset not found. Using California Housing instead...")
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv('data/california_housing.csv', index=False)
    print(f"✅ Loaded California Housing: {df.shape}")

# Show dataset info
print(f"\n📋 Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
print(f"📊 Target: {df.columns[-1]}")

# Initialize pipeline with feature selection
print("\n🤖 Running AutoClean...")
print("="*60)

pipeline = AutoClean(
    data=df,
    target=df.columns[-1],  # Use the last column as target
    test_size=0.2,
    random_state=42,
    scaling=True,
    scaling_method='robust',
    handle_missing=True,
    handle_outliers=True,
    remove_duplicates=True,
    drop_useless=True,
    feature_selection=True,
    selection_method='importance',
    top_k=15,
    verbose=True
)

# Run preprocessing
X_train, X_test, y_train, y_test = pipeline.preprocess()

# Results
print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Pipeline summary
pipeline.print_summary()

# Train a quick model
print("\n" + "="*70)
print("🤖 TRAINING MODEL")
print("="*70)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"\n📈 Training R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"📈 Test R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"📈 Test MAE: ${mean_absolute_error(y_test, y_pred_test):,.0f}")

# Feature importance
print("\n📊 Top 5 Most Important Features:")
importances = model.feature_importances_
imp_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False).head(5)

for _, row in imp_df.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

print("\n✅ Test complete!")