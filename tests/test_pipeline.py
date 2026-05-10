import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import AutoClean

# Load data
print("="*70)
print("🧪 TESTING FULL PIPELINE")
print("="*70)

# Initialize pipeline
pipeline = AutoClean(
    data='data/test_dataset.csv',
    target='rating',
    test_size=0.2,
    random_state=42,
    scaling=True,
    scaling_method='standard',
    verbose=True
)

# Run preprocessing
X_train, X_test, y_train, y_test = pipeline.preprocess()

print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Print summary
pipeline.print_summary()

# Test predict_ready_data
print("\n" + "="*70)
print("🔮 TESTING predict_ready_data")
print("="*70)
X_new = X_test.head(5)
X_new_processed = pipeline.predict_ready_data(X_new)
print(f"Input shape: {X_new.shape}")
print(f"Output shape: {X_new_processed.shape}")

# Test save/load
print("\n" + "="*70)
print("💾 TESTING SAVE/LOAD")
print("="*70)
pipeline.save('test_pipeline.pkl')
loaded = AutoClean.load('test_pipeline.pkl')
print("✅ Pipeline saved and loaded successfully")

print("\n✅ Full pipeline test complete!")