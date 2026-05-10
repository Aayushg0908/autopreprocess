# AutoPreprocess - Complete User Guide

## 📦 Installation

pip install autopreprocess-lite

## 🚀 Quick Start

from autopreprocess import AutoClean

# Load your data and preprocess in one line
pipeline = AutoClean('your_data.csv', target='target_column')
X_train, X_test, y_train, y_test = pipeline.preprocess()

# Train your model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.4f}")

## 📊 Complete Examples

### Example 1: Basic Usage

import pandas as pd
from autopreprocess import AutoClean

# Load data
df = pd.read_csv('house_prices.csv')

# Initialize pipeline
pipeline = AutoClean(
    data=df,
    target='SalePrice',
    test_size=0.2,
    random_state=42,
    verbose=True
)

# Preprocess
X_train, X_test, y_train, y_test = pipeline.preprocess()

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

### Example 2: With Feature Selection

pipeline = AutoClean(
    data='data.csv',
    target='price',
    feature_selection=True,
    selection_method='importance',  # 'variance', 'correlation', 'importance', 'mutual_info'
    top_k=20,
    verbose=True
)

X_train, X_test, y_train, y_test = pipeline.preprocess()

### Example 3: Custom Scaling Method

pipeline = AutoClean(
    data='data.csv',
    target='price',
    scaling=True,
    scaling_method='robust',  # 'standard', 'minmax', 'robust'
    verbose=True
)

### Example 4: Classification Problem

# AutoClean automatically detects classification vs regression
pipeline = AutoClean(
    data='titanic.csv',
    target='survived',  # Binary classification
    test_size=0.2,
    random_state=42
)

X_train, X_test, y_train, y_test = pipeline.preprocess()
# Stratified split is used automatically

### Example 5: Time Series Data

# For time series, no shuffling
pipeline = AutoClean(
    data='stock_prices.csv',
    target='price',
    test_size=0.2,
    random_state=None  # No shuffling
)

### Example 6: Save and Load Pipeline

# Save pipeline after preprocessing
pipeline.save('my_pipeline.pkl')

# Load for new predictions
loaded_pipeline = AutoClean.load('my_pipeline.pkl')

# Transform new data
X_new_clean = loaded_pipeline.predict_ready_data(X_new)

## 🔧 All Available Parameters

pipeline = AutoClean(
    # Required
    data='data.csv',           # CSV file path or pandas DataFrame
    target='target_column',    # Name of target column
    
    # Split Configuration
    test_size=0.2,             # Test size (default 0.2)
    random_state=42,           # Seed for reproducibility (default 42)
    
    # Preprocessing Steps
    scaling=True,              # Enable/disable scaling (default True)
    scaling_method='standard', # 'standard', 'minmax', or 'robust'
    handle_missing=True,       # Handle missing values (default True)
    handle_outliers=True,      # Handle outliers (default True)
    remove_duplicates=True,    # Remove duplicate rows (default True)
    drop_useless=True,         # Drop ID/constant columns (default True)
    
    # Feature Selection
    feature_selection=False,   # Enable feature selection (default False)
    selection_method='importance',  # 'variance', 'correlation', 'importance', 'mutual_info'
    top_k=20,                  # Number of features to keep
    
    # Column Control
    ignore_cols=['id', 'name'], # Columns to ignore (optional)
    
    # Logging
    verbose=True               # Print progress messages (default True)
)

## 📋 Available Methods

| Method | Description |
|--------|-------------|
| .preprocess() | Main method - splits, fits, transforms, returns train/test |
| .fit(X, y) | Fit pipeline on training data |
| .transform(X) | Transform new data using fitted pipeline |
| .fit_transform(X, y) | Fit and transform in one step |
| .predict_ready_data(X) | Transform new data for model predictions |
| .save(filepath) | Save pipeline to disk |
| .load(filepath) | Load pipeline from disk |
| .summary() | Get pipeline summary as dictionary |
| .print_summary() | Print readable pipeline summary |

## 📈 Feature Selection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| variance | Remove constant/low variance columns | Always (default first step) |
| correlation | Remove one from highly correlated pairs | Linear models |
| importance | Keep top K by RandomForest importance | Tree-based models |
| mutual_info | Keep top K by mutual information score | General purpose |

## 🔍 What AutoClean Does Automatically

1. Detects numeric, categorical, datetime, and useless columns
2. Splits data BEFORE any preprocessing (prevents data leakage)
3. Handles missing values based on percentage:
   - <5%: Fill with median/mode
   - 5-20%: Advanced imputation
   - 20-40%: Add missing indicator column
   - >40%: Drop column
4. Caps outliers at 1st and 99th percentiles
5. Encodes categorical columns:
   - <10 unique: One-hot encoding
   - >=10 unique: Frequency encoding
6. Scales numeric columns with your chosen method
7. Optionally selects top K features
8. Detects problem type (classification vs regression)
9. Uses stratified split for classification automatically

## 💡 Example Output

When verbose=True, you'll see:

======================================================================
🤖 AUTOCLEAN PIPELINE INITIALIZED
======================================================================
   Data shape: (1460, 81)
   Target: SalePrice
   Test size: 0.2

🚀 STARTING PREPROCESSING PIPELINE
======================================================================

📋 Step 0: Train/Test Split
   Train: (1168, 80), Test: (292, 80)

🔧 FITTING PIPELINE
========================================
   Problem type: regression

📋 Step 1: Cleaning
📋 Step 2: Encoding
📋 Step 3: Scaling
📋 Step 4: Feature Selection

======================================================================
✅ PREPROCESSING COMPLETE
======================================================================
   Total time: 0.61 seconds
   Train shape: (1168, 20)
   Test shape: (292, 20)

## 🎯 Real-World Example: Ames Housing

```python
from autopreprocess import AutoClean
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load and preprocess
pipeline = AutoClean(
    data='train.csv',
    target='SalePrice',
    test_size=0.2,
    feature_selection=True,
    selection_method='importance',
    top_k=20,
    verbose=True
)

X_train, X_test, y_train, y_test = pipeline.preprocess()

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
# Typical output: R² Score: 0.8163

📝 Notes
The import name is autopreprocess (not autopreprocess-lite)

Package name for pip is autopreprocess-lite

Hyphen in package name becomes underscore in import

Works with Python 3.8 and above

📄 License
MIT License

🤝 Contributing
Contributions welcome! Please submit issues and pull requests on GitHub.

⭐ Support
If you find this library useful, please star it on GitHub and share with others!

text
